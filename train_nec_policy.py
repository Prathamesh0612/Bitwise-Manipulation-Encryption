# train_nec_policy.py
# Standalone AI trainer for NEC Policy - Fixed version
import os, sys, math, argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

np.random.seed(42)
tf.random.set_seed(42)

def _entropy(data: bytes) -> float:
    if not data: return 0.0
    freq = np.zeros(256, dtype=np.float64)
    for b in data: freq[b] += 1
    p = freq / len(data)
    H = -(p[p>0] * np.log2(p[p>0])).sum()
    return float(H/8.0)

def _hist16(data: bytes, win: int = 65536) -> np.ndarray:
    data = data[:win]
    hist = np.zeros(16, dtype=np.float64)
    for b in data: hist[b // 16] += 1
    s = hist.sum() or 1.0
    return (hist/s).astype(np.float32)

def _magic_flags(header: bytes) -> np.ndarray:
    f = np.zeros(5, dtype=np.float32)
    if len(header) >= 4:
        if header[0]==0xFF and header[1]==0xD8: f[0]=1  # jpg
        elif header[0]==0x89 and header[1]==0x50 and header[2]==0x4E and header[3]==0x47: f[1]=1  # png
        elif header[0]==0x47 and header[1]==0x49 and header[2]==0x46: f[2]=1  # gif
        elif header[0]==0x25 and header[1]==0x50 and header[2]==0x44 and header[3]==0x46: f[3]=1  # pdf
        elif header[0]==0x50 and header[1]==0x4B: f[4]=1  # zip
    return f

def _size_bins(nbytes: int) -> np.ndarray:
    mb = nbytes / (1024*1024)
    if mb < 1:  return np.array([1,0,0], dtype=np.float32)
    if mb < 10: return np.array([0,1,0], dtype=np.float32)
    return np.array([0,0,1], dtype=np.float32)

def extract_features_from_file(path: str) -> np.ndarray:
    try:
        with open(path, 'rb') as f:
            data = f.read()
    except Exception:
        return np.zeros(25, dtype=np.float32)
    if len(data)==0:
        return np.zeros(25, dtype=np.float32)
    
    ent4  = _entropy(data[:4096])
    ent64 = _entropy(data[:65536])
    hist  = _hist16(data, 65536)
    flags = _magic_flags(data[:16])
    sizeb = _size_bins(len(data))
    
    feats = np.concatenate([[ent4], hist, flags, sizeb]).astype(np.float32)
    return feats  # shape (25,)

def process_dataset_dir(dataset_dir: str):
    X = []
    for root, _, files in os.walk(dataset_dir):
        for fn in files:
            fp = os.path.join(root, fn)
            X.append(extract_features_from_file(fp))
    if not X:
        raise RuntimeError(f"No files found under: {dataset_dir}")
    return np.stack(X, axis=0)

def weak_labels_from_feats(ent4: np.ndarray, ent64: np.ndarray, size_onehot: np.ndarray, magic_onehot: np.ndarray):
    avg_ent = 0.5*(ent4+ent64)
    ent_mult = np.where(avg_ent<0.3, 2.5, np.where(avg_ent<0.5, 2.0, np.where(avg_ent<0.7, 1.5, 1.0)))
    size_idx = size_onehot.argmax(axis=1)
    size_mult = np.where(size_idx==0, 1.5, np.where(size_idx==1, 1.2, 1.0))
    r = np.clip(0.005*ent_mult*size_mult, 0.001, 0.01).astype(np.float32).reshape(-1,1)
    
    has_magic = (magic_onehot.sum(axis=1) > 0).astype(np.float32)
    header = 0.33 + 0.07*has_magic
    structure = 0.33 + 0.07*has_magic
    random = 1.0 - (header+structure)
    w = np.stack([header, structure, random], axis=1).astype(np.float32)
    w = (w.T / (w.sum(axis=1)+1e-8)).T
    
    base_max = np.where(ent4>0.8, 8, np.where(ent4<0.3, 4, 6))
    min_scaled = np.zeros_like(base_max, dtype=np.float32)  # (2-2)/6
    max_scaled = ((base_max-2)/6.0).astype(np.float32)
    p = np.stack([min_scaled, max_scaled], axis=1)
    return r, w, p

def synth_features(N=30000):
    ent4  = np.clip(np.random.beta(2,2, size=N), 0, 1).astype(np.float32)
    ent64 = np.clip(ent4 + 0.1*np.random.randn(N), 0, 1).astype(np.float32)
    hist  = np.random.dirichlet(np.ones(16), size=N).astype(np.float32)
    magic_state = np.random.choice(6, size=N, p=[0.5,0.12,0.12,0.1,0.1,0.06])
    magic = np.zeros((N,5), dtype=np.float32)
    for i,s in enumerate(magic_state):
        if s>0: magic[i, s-1] = 1.0
    size_idx = np.random.choice(3, size=N, p=[0.4,0.5,0.1])
    sizeb = np.eye(3, dtype=np.float32)[size_idx]
    X = np.column_stack([ent4, hist, magic, sizeb]).astype(np.float32)
    return X, ent4, ent64, sizeb, magic

def build_model(d_in=25):
    x = keras.Input(shape=(d_in,), name='features')
    h = layers.Dense(64, activation='relu')(x)
    h = layers.Dense(64, activation='relu')(h)
    r = layers.Dense(1, activation='sigmoid', name='r')(h)
    w = layers.Dense(3, activation='softmax', name='w')(h)
    p = layers.Dense(2, activation='sigmoid', name='p')(h)
    m = keras.Model(x, [r,w,p])
    m.compile(optimizer='adam',
              loss={'r':'mse','w':'categorical_crossentropy','p':'mse'},
              loss_weights={'r':1.0,'w':1.0,'p':1.0},
              metrics={'r':'mae','w':'accuracy','p':'mae'})
    return m

def main():
    ap = argparse.ArgumentParser(description="NEC Policy AI trainer")
    ap.add_argument("--dataset", type=str, default=None, help="Path to dataset directory (optional)")
    ap.add_argument("--synth", action="store_true", help="Force synthetic training without real files")
    ap.add_argument("--out", type=str, default=".", help="Output directory for artifacts")
    args = ap.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    
    # Build training data
    if args.synth or not args.dataset:
        print("Using synthetic data (30k samples)")
        X, ent4, ent64, sizeb, magic = synth_features(N=30000)
    else:
        print(f"Processing real files from: {args.dataset}")
        X = process_dataset_dir(args.dataset)
        ent4  = X[:, 0]
        hist  = X[:, 1:17]
        magic = X[:, 17:22]
        sizeb = X[:, 22:25]
        spread = np.abs(hist - hist.mean(axis=1, keepdims=True)).mean(axis=1)
        ent64 = np.clip(ent4 + 0.2*(0.5 - spread), 0, 1).astype(np.float32)
    
    y_r, y_w, y_p = weak_labels_from_feats(ent4, ent64, sizeb, magic)
    
    # Train/val split
    idx = np.arange(X.shape[0]); np.random.shuffle(idx)
    ntr = int(0.8*len(idx))
    tr, va = idx[:ntr], idx[ntr:]
    
    print(f"Training samples: {len(tr)}, Validation: {len(va)}")
    
    model = build_model(25)
    print("Model created:")
    model.summary()
    
    cb = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    ]
    
    print("Starting training...")
    history = model.fit(
        X[tr], {'r':y_r[tr], 'w':y_w[tr], 'p':y_p[tr]},
        validation_data=(X[va], {'r':y_r[va], 'w':y_w[va], 'p':y_p[va]}),
        batch_size=256, epochs=40, callbacks=cb, verbose=1
    )
    
    # Save artifacts
    out_savedmodel = os.path.join(args.out, "nec_policy_savedmodel")
    out_h5         = os.path.join(args.out, "nec_policy_keras.h5")
    
    try:
        model.export(out_savedmodel)
        print(f"✓ SavedModel exported: {out_savedmodel}")
    except Exception as e:
        print(f"⚠ SavedModel export failed: {e}")
    
    try:
        model.save(out_h5)
        print(f"✓ H5 model saved: {out_h5}")
    except Exception as e:
        print(f"⚠ H5 save failed: {e}")
    
    # Validation
    pr = model.predict(X[va][:5])
    r_scaled = 0.001 + 0.009*pr[0].reshape(-1)
    print("\nSample predictions (r scaled):", [f"{float(v):.5f}" for v in r_scaled])
    
    print("\nTraining completed!")
    print(f"Next step: Run './export_tfjs.sh' to convert to TensorFlow.js")

if __name__ == "__main__":
    main()

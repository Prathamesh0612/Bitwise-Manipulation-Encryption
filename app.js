// NEC - Neural Encryption Corrector - AI-Enhanced Single-Key System
// Client-side implementation with Web Crypto API and TensorFlow.js
let necModel = null;

// Load NEC AI model
async function loadNECModel() {
    try {
        necModel = await tf.loadLayersModel('./');
        console.log('✓ NEC AI model loaded successfully');
        updateAIStatus('✓ AI Model Active');
        return true;
    } catch (e) {
        console.log('⚠ NEC AI model not found, using heuristic fallback');
        necModel = null;
        updateAIStatus('100% Safe');
        return false;
    }
}

function updateAIStatus(status) {
    const statusEl = document.getElementById('ai-status');
    if (statusEl) statusEl.textContent = status;
}

// Initialize app after model loading
document.addEventListener('DOMContentLoaded', async () => {
    await loadNECModel();
    new NECApp();
});

class NECrypto {
    constructor() {
        this.VERSION = 1;
        this.KDF_ITERATIONS = 150000;
        this.SALT_SIZE = 32;
        this.SEED_SIZE = 32;
        this.MAX_FILE_SIZE = 150 * 1024 * 1024;
        this.CHUNK_SIZE = 10 * 1024 * 1024;
        this.BASES = [12, 16, 20, 36];
        this.MIN_CORRUPTION_RATIO = 0.001;
        this.MAX_CORRUPTION_RATIO = 0.01;
        this.MAX_CRYPTO_RANDOM_BYTES = 65536;
    }

    generateRandomBytes(size) {
        if (size <= this.MAX_CRYPTO_RANDOM_BYTES) {
            const array = new Uint8Array(size);
            crypto.getRandomValues(array);
            return array;
        }
        const result = new Uint8Array(size);
        let offset = 0;
        while (offset < size) {
            const chunkSize = Math.min(this.MAX_CRYPTO_RANDOM_BYTES, size - offset);
            const chunk = new Uint8Array(chunkSize);
            crypto.getRandomValues(chunk);
            result.set(chunk, offset);
            offset += chunkSize;
        }
        return result;
    }

    csprngInt(maxExclusive) {
        if (maxExclusive <= 1) return 0;
        const u32 = new Uint32Array(1);
        const limit = Math.floor(0x100000000 / maxExclusive) * maxExclusive;
        let x;
        do { crypto.getRandomValues(u32); x = u32[0]; } while (x >= limit);
        return x % maxExclusive;
    }

    generateKeyString() {
        const keyBytes = this.generateRandomBytes(32);
        return this.base85Encode(keyBytes);
    }

    bufferToHex(buffer) {
        return Array.from(new Uint8Array(buffer)).map(b => b.toString(16).padStart(2, '0')).join('');
    }

    hexToBuffer(hex) {
        const bytes = new Uint8Array(hex.length / 2);
        for (let i = 0; i < hex.length; i += 2) bytes[i / 2] = parseInt(hex.substr(i, 2), 16);
        return bytes.buffer;
    }

    async sha256(data) {
        const buffer = typeof data === 'string' ? new TextEncoder().encode(data) : data;
        const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
        return this.bufferToHex(hashBuffer);
    }

    async deriveKeysFromKeyString(keyString, salt, iterations = this.KDF_ITERATIONS) {
        const keyStringBuffer = new TextEncoder().encode(keyString);
        const saltBuffer = (salt instanceof Uint8Array) ? salt : new Uint8Array(salt);
        const keyMaterial = await crypto.subtle.importKey('raw', keyStringBuffer, 'PBKDF2', false, ['deriveBits']);
        const derivedBits = await crypto.subtle.deriveBits(
            { name: 'PBKDF2', salt: saltBuffer, iterations, hash: 'SHA-256' },
            keyMaterial,
            512
        );
        const d = new Uint8Array(derivedBits);
        return { masterKey: d.slice(0, 32), macKey: d.slice(32, 64) };
    }

    xorEncryptDecrypt(data, key) {
        const result = new Uint8Array(data.length);
        for (let i = 0; i < data.length; i++) result[i] = data[i] ^ key[i % key.length];
        return result;
    }

    async computeHMAC(key, data) {
        const keyBuffer = (key instanceof Uint8Array) ? key : new TextEncoder().encode(key);
        const dataBuffer = (data instanceof Uint8Array) ? data : new TextEncoder().encode(data);
        const cryptoKey = await crypto.subtle.importKey('raw', keyBuffer, { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
        const signature = await crypto.subtle.sign('HMAC', cryptoKey, dataBuffer);
        return new Uint8Array(signature);
    }

    async verifyHMAC(key, data, expectedTag) {
        const computedTag = await this.computeHMAC(key, data);
        return this.constantTimeEquals(computedTag, expectedTag);
    }

    constantTimeEquals(a, b) {
        if (a.length !== b.length) return false;
        let r = 0; for (let i = 0; i < a.length; i++) r |= a[i] ^ b[i];
        return r === 0;
    }

    async generatePRFStream(seed, context, length) {
        let output = new Uint8Array(0);
        let counter = 0;
        while (output.length < length) {
            const ctx = new Uint8Array(context.length + 4);
            ctx.set(context);
            ctx.set(new Uint8Array([
                (counter >> 24) & 0xff, (counter >> 16) & 0xff, (counter >> 8) & 0xff, counter & 0xff
            ]), context.length);
            const chunk = await this.computeHMAC(seed, ctx);
            const next = new Uint8Array(output.length + chunk.length);
            next.set(output); next.set(chunk, output.length);
            output = next;
            counter++;
        }
        return output.slice(0, length);
    }

    generateRandomPartition(n, minParts = 2, maxParts = 8) {
        const parts = Math.min(maxParts, Math.max(minParts, Math.min(n, 8)));
        const partition = new Array(parts).fill(1);
        let remaining = Math.max(0, n - parts);
        while (remaining > 0) {
            const idx = this.csprngInt(parts);
            const add = Math.min(remaining, 1 + this.csprngInt(10));
            partition[idx] += add;
            remaining -= add;
        }
        for (let i = partition.length - 1; i > 0; i--) {
            const j = this.csprngInt(i + 1);
            [partition[i], partition[j]] = [partition[j], partition[i]];
        }
        return partition;
    }

    async generateBitPositions(seed, fileHashHex, chunkId, partition, totalBits) {
        const allPositions = [];
        const fileHashBuf = new Uint8Array(this.hexToBuffer(fileHashHex));
        for (let partId = 0; partId < partition.length; partId++) {
            const context = new Uint8Array(32 + 4 + 4);
            context.set(fileHashBuf, 0);
            context.set(new Uint8Array([
                (chunkId >> 24) & 0xff, (chunkId >> 16) & 0xff, (chunkId >> 8) & 0xff, chunkId & 0xff
            ]), 32);
            context.set(new Uint8Array([
                (partId >> 24) & 0xff, (partId >> 16) & 0xff, (partId >> 8) & 0xff, partId & 0xff
            ]), 36);

            const stream = await this.generatePRFStream(seed, context, partition[partId] * 4 + 16);
            const positions = new Set();
            for (let i = 0; i + 3 < stream.length && positions.size < partition[partId]; i += 4) {
                const value = (stream[i] << 24) | (stream[i + 1] << 16) | (stream[i + 2] << 8) | stream[i + 3];
                const pos = Math.abs(value) % totalBits;
                positions.add(pos);
            }
            while (positions.size < partition[partId]) {
                const rb = this.generateRandomBytes(4);
                const value = (rb[0] << 24) | (rb[1] << 16) | (rb[2] << 8) | rb[3];
                positions.add(Math.abs(value) % totalBits);
            }
            allPositions.push(...Array.from(positions));
        }
        return allPositions;
    }

    flipBits(data, positions) {
        const result = new Uint8Array(data);
        for (const position of positions) {
            const byteIndex = Math.floor(position / 8);
            const bitIndex = position % 8;
            if (byteIndex < result.length) result[byteIndex] ^= (1 << (7 - bitIndex));
        }
        return result;
    }

    base85Encode(data) {
        let binary = ''; for (let i = 0; i < data.length; i++) binary += String.fromCharCode(data[i]);
        return btoa(binary);
    }

    base85Decode(str) {
        const binary = atob(str);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
        return bytes;
    }

    async createCompactKey(keyString, salt, iterations, partition, bases, fileHashHex, seed) {
        const { masterKey, macKey } = await this.deriveKeysFromKeyString(keyString, salt, iterations);
        const encryptedSeed = this.xorEncryptDecrypt(seed, masterKey);
        const header = new Uint8Array(4 + 32 + 4 + 32 + 1 + partition.length * 2 + 1 + bases.length + 32);
        let off = 0;
        header.set(new Uint8Array([0, 0, 0, this.VERSION]), off); off += 4;
        header.set(salt, off); off += 32;
        header.set(new Uint8Array([(iterations >> 24) & 0xff, (iterations >> 16) & 0xff, (iterations >> 8) & 0xff, iterations & 0xff]), off); off += 4;
        header.set(encryptedSeed, off); off += 32;
        header[off++] = partition.length;
        for (const p of partition) { header.set(new Uint8Array([(p >> 8) & 0xff, p & 0xff]), off); off += 2; }
        header[off++] = bases.length;
        for (const b of bases) header[off++] = b;
        header.set(new Uint8Array(this.hexToBuffer(fileHashHex)), off);

        const tag = await this.computeHMAC(macKey, header);
        const combined = new Uint8Array(header.length + tag.length);
        combined.set(header, 0); combined.set(tag, header.length);

        const base = this.base85Encode(combined);
        const checksum = (await this.sha256(base)).slice(0, 8);
        return `NEC${checksum}${base}`;
    }

    async parseCompactKeyFromHeader(headerString, keyString) {
        if (!headerString.startsWith('NEC')) throw new Error('Invalid key header format');
        const checksum = headerString.substring(3, 11);
        const base = headerString.substring(11);
        const computed = (await this.sha256(base)).slice(0, 8);
        if (checksum !== computed) throw new Error('Key checksum mismatch');

        const combined = this.base85Decode(base);
        const headerLen = combined.length - 32;
        const header = combined.slice(0, headerLen);
        const expectedTag = combined.slice(headerLen);

        let off = 0;
        const version = (header[off] << 24) | (header[off + 1] << 16) | (header[off + 2] << 8) | header[off + 3]; off += 4;
        const salt = header.slice(off, off + 32); off += 32;
        const iterations = (header[off] << 24) | (header[off + 1] << 16) | (header[off + 2] << 8) | header[off + 3]; off += 4;
        const encryptedSeed = header.slice(off, off + 32); off += 32;
        const partLen = header[off++]; const partition = [];
        for (let i = 0; i < partLen; i++) { partition.push((header[off] << 8) | header[off + 1]); off += 2; }
        const baseLen = header[off++]; const bases = [];
        for (let i = 0; i < baseLen; i++) bases.push(header[off++]);
        const fileHashHex = this.bufferToHex(header.slice(off, off + 32));

        const { masterKey, macKey } = await this.deriveKeysFromKeyString(keyString, salt, iterations);
        const ok = await this.verifyHMAC(macKey, header, expectedTag);
        if (!ok) throw new Error('Key verification failed - invalid key');

        const seed = this.xorEncryptDecrypt(encryptedSeed, masterKey);
        return { version, salt, iterations, partition, bases, fileHash: fileHashHex, seed, macKey };
    }

    // AI-enhanced file analysis
    buildAIFeatures(bytes) {
        const features = [];
        const entropy4k = this.calculateEntropy(bytes.slice(0, Math.min(4096, bytes.length)));
        const entropy64k = this.calculateEntropy(bytes.slice(0, Math.min(65536, bytes.length)));
        features.push(entropy4k, entropy64k);

        const hist = new Array(16).fill(0);
        const sample = bytes.slice(0, Math.min(65536, bytes.length));
        for (const b of sample) hist[Math.floor(b / 16)]++;
        const sum = hist.reduce((a,b) => a+b, 0) || 1;
        features.push(...hist.map(v => v/sum));

        const h = bytes.slice(0, 16);
        const flags = [
            (h[0]===0xFF && h[1]===0xD8),
            (h[0]===0x89 && h[1]===0x50 && h[2]===0x4E && h[3]===0x47),
            (h[0]===0x47 && h[1]===0x49 && h[2]===0x46),
            (h[0]===0x25 && h[1]===0x50 && h[2]===0x44 && h[3]===0x46),
            (h[0]===0x50 && h[1]===0x4B)
        ].map(b => b ? 1 : 0);
        features.push(...flags);

        const sizeMB = bytes.length / (1024 * 1024);
        const sizeBins = [sizeMB < 1, sizeMB >= 1 && sizeMB < 10, sizeMB >= 10].map(b => b ? 1 : 0);
        features.push(...sizeBins);

        return features;
    }

    analyzeFile(data) {
        const bytes = new Uint8Array(data);

        if (typeof tf !== 'undefined' && necModel) {
            try {
                const features = this.buildAIFeatures(bytes);
                const x = tf.tensor(features, [1, features.length]);
                const predictions = necModel.predict(x);

                const r_raw = predictions[0].dataSync()[0];
                const w_logits = Array.from(predictions[1].dataSync());
                const p_raw = Array.from(predictions[2].dataSync());

                const r = Math.max(0.001, Math.min(0.01, 0.001 + 0.009 * r_raw));
                const w_sum = w_logits.reduce((a,b) => a+Math.exp(b), 0);
                const w = w_logits.map(v => Math.exp(v)/w_sum);
                const minP = Math.max(2, Math.min(8, Math.round(2 + 6 * p_raw[0])));
                const maxP = Math.max(minP, Math.min(8, Math.round(2 + 6 * p_raw[1])));

                x.dispose();
                if (Array.isArray(predictions)) predictions.forEach(p => p.dispose());

                return {
                    size: bytes.length,
                    entropy: this.calculateEntropy(bytes.slice(0, Math.min(1024 * 1024, bytes.length))),
                    fileType: this.detectFileType(bytes),
                    corruptionRatio: r,
                    partitionCount: Math.round((minP + maxP) / 2),
                    bases: this.BASES,
                    strategyWeights: { header: w[0], structure: w[1], random: w[2] },
                    aiUsed: true
                };
            } catch (e) {
                console.warn('AI prediction failed, using heuristic:', e);
            }
        }

        // Heuristic fallback
        const analysis = {
            size: bytes.length,
            entropy: this.calculateEntropy(bytes.slice(0, Math.min(1024 * 1024, bytes.length))),
            fileType: this.detectFileType(bytes),
            corruptionRatio: 0,
            partitionCount: 0,
            bases: [],
            aiUsed: false
        };

        // Use a crypto-safe random float [0, 1] for randomization
        // We get 4 random bytes and divide by the max possible 4-byte value
        const randomVal = (this.generateRandomBytes(4).reduce((a, b) => a * 256 + b, 0)) / 0xFFFFFFFF;

        // 1. Random Corruption Ratio (still influenced by entropy)
        //    Base ratio: 0.001 (low entropy) to 0.008 (high entropy)
        const baseRatio = this.MIN_CORRUPTION_RATIO + (analysis.entropy * 0.007);
        //    Final ratio: Add randomness, but clamp to [MIN, MAX]
        analysis.corruptionRatio = Math.max(this.MIN_CORRUPTION_RATIO, Math.min(this.MAX_CORRUPTION_RATIO,
            baseRatio + (randomVal - 0.5) * 0.002 // Add/subtract up to 0.001
        ));
        
        // 2. Random Partition Count
        //    Use csprngInt to get a random int from 2 to 8
        analysis.partitionCount = 2 + this.csprngInt(7); // 2 + (random 0-6) = 2 to 8

        // 3. Random Bases
        //    Randomly pick 2, 3, or 4 bases from the list
        const numBases = 2 + this.csprngInt(3); // 2, 3, or 4
        // Create a copy, shuffle it, and take the first numBases
        const shuffledBases = [...this.BASES].sort(() => 0.5 - Math.random());
        analysis.bases = shuffledBases.slice(0, numBases);

        // --- END MODIFICATION ---

        return analysis;
    }
    }

    calculateEntropy(bytes) {
        const freq = new Array(256).fill(0);
        for (const b of bytes) freq[b]++;
        let H = 0, n = bytes.length;
        for (const c of freq) if (c > 0) { const p = c / n; H -= p * Math.log2(p); }
        return H / 8;
    }

    detectFileType(bytes) {
        const h = bytes.slice(0, 16);
        if (h[0] === 0xFF && h[1] === 0xD8) return 'image/jpeg';
        if (h[0] === 0x89 && h[1] === 0x50 && h[2] === 0x4E && h[3] === 0x47) return 'image/png';
        if (h[0] === 0x47 && h[1] === 0x49 && h[2] === 0x46) return 'image/gif';
        if (h[0] === 0x25 && h[1] === 0x50 && h[2] === 0x44 && h[3] === 0x46) return 'application/pdf';
        if (h[0] === 0x50 && h[1] === 0x4B) return 'application/zip';
        let textScore = 0;
        for (let i = 0; i < Math.min(1024, bytes.length); i++) {
            const b = bytes[i];
            if ((b >= 32 && b <= 126) || b === 9 || b === 10 || b === 13) textScore++;
        }
        if (textScore > bytes.length * 0.8) return 'text/plain';
        return 'application/octet-stream';
    }

    generateTestData(size = 1024 * 1024) { return this.generateRandomBytes(size); }
}

class NECApp {
    constructor() {
        this.crypto = new NECrypto();
        this.currentFile = null;
        this.encryptedData = null;
        this.restoredData = null;
        this.originalFileName = null;
        this.initializeUI();
    }

    initializeUI() {
      // Screenshot button functionality
const screenshotBtn = document.getElementById('take-screenshot');
if (screenshotBtn) {
    screenshotBtn.addEventListener('click', async () => {
        try {
            const element = document.getElementById('encrypt-results');
            if (!element) {
                alert('No encryption results to capture');
                return;
            }
            const canvas = await html2canvas(element, { scale: 2 });
            canvas.toBlob(blob => {
                const a = document.createElement('a');
                a.href = URL.createObjectURL(blob);
                a.download = `NecEncryptionResults_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.png`;
                a.click();
                URL.revokeObjectURL(a.href);
            });
        } catch (e) {
            alert('Screenshot failed: ' + e.message);
        }
    });
}

// Copy Key button functionality
const copyKeyBtn = document.querySelector('.copy-btn[data-target="encryption-key"]');
if (copyKeyBtn) {
    copyKeyBtn.addEventListener('click', () => {
        const keyArea = document.getElementById('encryption-key');
        if (!keyArea) return;
        keyArea.select();
        document.execCommand('copy');
        copyKeyBtn.textContent = 'Copied!';
        setTimeout(() => {
            copyKeyBtn.innerHTML = '<i class="fas fa-copy"></i> Copy';
        }, 3000);
    });
}

// Save Key as TXT button functionality
const saveKeyBtn = document.getElementById('save-key-txt');
if (saveKeyBtn) {
    saveKeyBtn.addEventListener('click', () => {
        const keyArea = document.getElementById('encryption-key');
        if (!keyArea || !keyArea.value) {
            alert('No encryption key to save');
            return;
        }
        const blob = new Blob([keyArea.value], { type: 'text/plain' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'NEC_Encryption_Key_' + new Date().toISOString().slice(0, 19).replace(/:/g, '-') + '.txt';
        a.click();
        URL.revokeObjectURL(a.href);
    });
}

        document.querySelectorAll('.tab-btn').forEach(btn => 
            btn.addEventListener('click', (e) => this.switchTab(e.target.dataset.tab))
        );
        this.setupFileUpload('encrypt');
        this.setupFileUpload('decrypt');
        document.getElementById('start-encrypt').addEventListener('click', () => this.encryptFile());
        document.getElementById('start-decrypt').addEventListener('click', () => this.decryptFile());
        document.getElementById('run-self-test').addEventListener('click', () => this.runSelfTest());
        document.querySelectorAll('.copy-btn').forEach(btn => 
            btn.addEventListener('click', (e) => this.copyToClipboard(e.target.dataset.copy))
        );
        document.getElementById('download-encrypted').addEventListener('click', () => 
            this.downloadFile(this.encryptedData, `${this.currentFile?.name || 'encrypted'}.bme`)
        );
        document.getElementById('download-restored').addEventListener('click', () => 
            this.downloadFile(this.restoredData, this.originalFileName || 'restored.bin')
        );
        this.createDemoFileButton();
    }

    createDemoFileButton() {
        const uploadArea = document.getElementById('encrypt-upload');
        const demoButton = document.createElement('button');
        demoButton.className = 'btn btn--outline demo-btn';
        demoButton.textContent = 'Use Demo File (1KB Text)';
        demoButton.style.marginTop = '1rem';
        demoButton.addEventListener('click', () => {
            const demoText = 'This is a demo file for testing NEC AI-enhanced encryption system.\n'.repeat(25);
            const demoFile = new File([demoText], 'demo.txt', { type: 'text/plain' });
            this.handleFileSelect(demoFile, 'encrypt');
        });
        uploadArea.appendChild(demoButton);
    }

    switchTab(tab) {
        document.querySelectorAll('.tab-btn').forEach(btn => 
            btn.classList.toggle('active', btn.dataset.tab === tab)
        );
        document.querySelectorAll('.tab-content').forEach(c => 
            c.classList.toggle('active', c.id === `${tab}-tab`)
        );
    }

    setupFileUpload(type) {
        const area = document.getElementById(`${type}-upload`);
        const input = document.getElementById(`${type}-file`);
        
        area.addEventListener('click', (e) => { 
            if (e.target.tagName !== 'BUTTON') input.click(); 
        });
        area.addEventListener('dragover', (e) => { 
            e.preventDefault(); area.classList.add('dragover'); 
        });
        area.addEventListener('dragleave', (e) => { 
            e.preventDefault(); area.classList.remove('dragover'); 
        });
        area.addEventListener('drop', (e) => { 
            e.preventDefault(); 
            area.classList.remove('dragover'); 
            const files = e.dataTransfer.files; 
            if (files.length) this.handleFileSelect(files[0], type); 
        });
        input.addEventListener('change', (e) => { 
            if (e.target.files && e.target.files.length) this.handleFileSelect(e.target.files[0], type); 
        });
    }

    async handleFileSelect(file, type) {
        if (file.size > this.crypto.MAX_FILE_SIZE) {
            this.showError(`File too large. Maximum size is ${this.crypto.MAX_FILE_SIZE / (1024 * 1024)} MB`);
            return;
        }
        
        try {
            if (type === 'encrypt') {
                this.currentFile = file;
                await this.analyzeFile(file);
                document.getElementById('start-encrypt').classList.remove('hidden');
            } else {
                this.encryptedFile = file;
                document.getElementById('decrypt-credentials').classList.remove('hidden');
                document.getElementById('start-decrypt').classList.remove('hidden');
            }
        } catch (err) {
            this.showError(`File processing error: ${err.message}`);
        }
    }

    async analyzeFile(file) {
        const data = await file.arrayBuffer();
        const analysis = this.crypto.analyzeFile(data);
        
        document.getElementById('encrypt-filename').textContent = file.name;
        document.getElementById('encrypt-filesize').textContent = this.formatFileSize(file.size);
        document.getElementById('encrypt-filetype').textContent = analysis.fileType;
        document.getElementById('encrypt-entropy').textContent = analysis.entropy.toFixed(3);
        document.getElementById('encrypt-ratio').textContent = `${(analysis.corruptionRatio * 100).toFixed(3)}%`;
        document.getElementById('encrypt-partitions').textContent = analysis.partitionCount;
        
        // Update AI status
        const aiStatus = analysis.aiUsed ? '✓ AI Enhanced' : '⚠ Heuristic Mode';
        updateAIStatus(aiStatus);
        
        document.getElementById('encrypt-info').classList.remove('hidden');
        this.fileAnalysis = analysis;
    }

    showProgress(type, percent) {
        const fill = document.getElementById(`${type}-progress-fill`);
        const text = document.getElementById(`${type}-progress-text`);
        const bar = document.getElementById(`${type}-progress`);
        bar.classList.remove('hidden');
        fill.style.width = `${percent}%`;
        text.textContent = `${Math.round(percent)}%`;
    }

    showError(message) { this.showMessage(message, 'error'); }
    showSuccess(message) { this.showMessage(message, 'success'); }
    
    showMessage(message, type) {
        document.querySelectorAll('.error-message, .success-message').forEach(el => el.remove());
        const div = document.createElement('div');
        div.className = `${type}-message fade-in`;
        div.textContent = message;
        div.style.cssText = `
            padding: 12px; margin: 12px 0; border-radius: 6px; font-size: 14px;
            background: rgba(${type === 'error' ? '192, 21, 47' : '33, 128, 141'}, 0.1);
            border: 1px solid rgba(${type === 'error' ? '192, 21, 47' : '33, 128, 141'}, 0.3);
            color: var(--color-${type === 'error' ? 'error' : 'success'});
        `;
        const active = document.querySelector('.tab-content.active');
        active.insertBefore(div, active.firstChild);
        setTimeout(() => { if (div.parentNode) div.parentNode.removeChild(div); }, 5000);
    }

    async copyToClipboard(elementId) {
        const el = document.getElementById(elementId);
        const btn = document.querySelector(`[data-copy="${elementId}"]`);
        try {
            await navigator.clipboard.writeText(el.value);
            const orig = btn.textContent;
            btn.textContent = 'Copied!';
            setTimeout(() => btn.textContent = orig, 2000);
        } catch {
            el.select();
            document.execCommand('copy');
            const orig = btn.textContent;
            btn.textContent = 'Copied!';
            setTimeout(() => btn.textContent = orig, 2000);
        }
    }

    async encryptFile() {
        if (!this.currentFile) return;
        const t0 = Date.now();
        this.showProgress('encrypt', 0);
        
        try {
            const salt = this.crypto.generateRandomBytes(32);
            const seed = this.crypto.generateRandomBytes(32);
            const keyString = this.crypto.generateKeyString();

            const data = new Uint8Array(await this.currentFile.arrayBuffer());
            const fileHash = await this.crypto.sha256(data);
            document.getElementById('original-hash').textContent = fileHash;

            const totalBits = data.length * 8;
            const partition = this.crypto.generateRandomPartition(
                Math.floor(totalBits * this.fileAnalysis.corruptionRatio), 2, 8
            );
            const bases = this.fileAnalysis.bases;

            this.showProgress('encrypt', 25);

            const encryptedChunks = [];
            const cs = this.crypto.CHUNK_SIZE;
            for (let i = 0; i < data.length; i += cs) {
                const chunk = data.slice(i, Math.min(i + cs, data.length));
                const chunkId = Math.floor(i / cs);
                const positions = await this.crypto.generateBitPositions(seed, fileHash, chunkId, partition, chunk.length * 8);
                encryptedChunks.push(this.crypto.flipBits(chunk, positions));
                this.showProgress('encrypt', 25 + (i / data.length) * 50);
            }
            
            const encryptedData = new Uint8Array(data.length);
            let off = 0;
            for (const c of encryptedChunks) {
                encryptedData.set(c, off);
                off += c.length;
            }

            this.showProgress('encrypt', 75);
            const encryptedHash = await this.crypto.sha256(encryptedData);
            document.getElementById('encrypted-hash').textContent = encryptedHash;

            const headerString = await this.crypto.createCompactKey(
                keyString, salt, this.crypto.KDF_ITERATIONS, partition, bases, fileHash, seed
            );

            const preface = `NECHDR\n${headerString}\nENDHDR\n`;
            const prefaceBytes = new TextEncoder().encode(preface);
            const finalData = new Uint8Array(prefaceBytes.length + encryptedData.length);
            finalData.set(prefaceBytes, 0);
            finalData.set(encryptedData, prefaceBytes.length);
            this.encryptedData = finalData;

            this.showProgress('encrypt', 100);

            document.getElementById('encryption-key').value = keyString;
            document.getElementById('encrypt-results').classList.remove('hidden');

            const t1 = Date.now();
            const thr = (data.length / ((t1 - t0) / 1000)) / (1024 * 1024);
            document.getElementById('encrypt-throughput').textContent = `${thr.toFixed(1)} MB/s`;
            
            const aiNote = this.fileAnalysis.aiUsed ? ' (AI-Enhanced)' : ' (Heuristic Mode)';
            this.showSuccess(`File encrypted successfully${aiNote}! Save your key and the .nec file.`);
        } catch (err) {
            this.showError(`Encryption failed: ${err.message}`);
        }
    }

    async decryptFile() {
        if (!this.encryptedFile) return;
        const keyString = document.getElementById('decrypt-key').value.trim();
        if (!keyString) {
            this.showError('Please provide the encryption key');
            return;
        }

        const t0 = Date.now();
        this.showProgress('decrypt', 0);
        
        try {
            const raw = new Uint8Array(await this.encryptedFile.arrayBuffer());

            const scanLen = Math.min(65536, raw.length);
            const headText = new TextDecoder().decode(raw.slice(0, scanLen));
            const startMarker = 'NECHDR\n';
            const endMarker = '\nENDHDR\n';
            
            if (!headText.startsWith(startMarker)) throw new Error('Missing key header in file');
            const endIdx = headText.indexOf(endMarker);
            if (endIdx < 0) throw new Error('Corrupted key header in file');
            
            const headerString = headText.substring(startMarker.length, endIdx);
            const headerBytesLen = new TextEncoder().encode(headText.substring(0, endIdx + endMarker.length)).length;

            const keyData = await this.crypto.parseCompactKeyFromHeader(headerString, keyString);
            this.showProgress('decrypt', 25);

            const encryptedData = raw.slice(headerBytesLen);

            const decryptedChunks = [];
            const cs = this.crypto.CHUNK_SIZE;
            for (let i = 0; i < encryptedData.length; i += cs) {
                const chunk = encryptedData.slice(i, Math.min(i + cs, encryptedData.length));
                const chunkId = Math.floor(i / cs);
                const positions = await this.crypto.generateBitPositions(
                    keyData.seed, keyData.fileHash, chunkId, keyData.partition, chunk.length * 8
                );
                decryptedChunks.push(this.crypto.flipBits(chunk, positions));
                this.showProgress('decrypt', 25 + (i / encryptedData.length) * 60);
            }
            
            const restored = new Uint8Array(encryptedData.length);
            let off = 0;
            for (const c of decryptedChunks) {
                restored.set(c, off);
                off += c.length;
            }

            this.showProgress('decrypt', 80);
            const restoredHash = await this.crypto.sha256(restored);
            document.getElementById('verify-original-hash').textContent = keyData.fileHash;
            document.getElementById('verify-restored-hash').textContent = restoredHash;

            const ok = (restoredHash === keyData.fileHash);
            const statusEl = document.getElementById('hash-match-status');
            statusEl.textContent = ok ? 'MATCH ✓' : 'MISMATCH ✗';
            statusEl.className = ok ? 'status status--success' : 'status status--error';
            
            document.getElementById('decrypt-verification').classList.remove('hidden');
            this.showProgress('decrypt', 100);

            if (ok) this.showSuccess('File decrypted and verified successfully!');
            else this.showError('Hash verification failed - wrong key or corrupted file');

            this.restoredData = restored;
            this.originalFileName = this.encryptedFile.name.replace(/\.bme$/i, '') || 'restored.bin';


            // FIXED: Make download button visible and functional
            const dlBtn = document.getElementById('download-restored');
            if (dlBtn) {
                dlBtn.classList.remove('hidden');
                dlBtn.removeAttribute('disabled');
                dlBtn.textContent = `Download Restored File`;
            }

            // Fallback link
            const link = document.getElementById('download-restored-link');
            if (link && this.restoredData) {
                const blob = new Blob([this.restoredData], { type: 'application/octet-stream' });
                const url = URL.createObjectURL(blob);
                link.href = url;
                link.download = this.originalFileName;
                link.classList.remove('hidden');
                link.addEventListener('click', () => setTimeout(() => URL.revokeObjectURL(url), 1000), { once: true });
            }

            const t1 = Date.now();
            const thr = (restored.length / ((t1 - t0) / 1000)) / (1024 * 1024);
            document.getElementById('decrypt-throughput').textContent = `${thr.toFixed(1)} MB/s`;
        } catch (err) {
            this.showError(`Decryption failed: ${err.message}`);
        }
    }

    async runSelfTest() {
        const resultsDiv = document.getElementById('self-test-results');
        const statusDiv = resultsDiv.querySelector('.test-status');
        const detailsDiv = resultsDiv.querySelector('.test-details');
        resultsDiv.classList.remove('hidden');
        statusDiv.textContent = 'Running self-test...';
        statusDiv.className = 'test-status pulse';
        
        try {
            const testData = this.crypto.generateTestData(1024 * 1024);
            const originalHash = await this.crypto.sha256(testData);
            const salt = this.crypto.generateRandomBytes(32);
            const seed = this.crypto.generateRandomBytes(32);
            const keyString = this.crypto.generateKeyString();
            const partition = this.crypto.generateRandomPartition(1000, 3, 5);
            const bases = [12, 16, 20];

            const positions = await this.crypto.generateBitPositions(seed, originalHash, 0, partition, testData.length * 8);
            const encryptedData = this.crypto.flipBits(testData, positions);

            const headerString = await this.crypto.createCompactKey(
                keyString, salt, this.crypto.KDF_ITERATIONS, partition, bases, originalHash, seed
            );
            const keyData = await this.crypto.parseCompactKeyFromHeader(headerString, keyString);

            const decryptPositions = await this.crypto.generateBitPositions(
                keyData.seed, keyData.fileHash, 0, keyData.partition, encryptedData.length * 8
            );
            const restoredData = this.crypto.flipBits(encryptedData, decryptPositions);

            const restoredHash = await this.crypto.sha256(restoredData);
            const hashMatch = (originalHash === restoredHash);
            const seedMatch = this.crypto.constantTimeEquals(seed, keyData.seed);
            const dataMatch = this.crypto.constantTimeEquals(testData, restoredData);

            const aiStatus = necModel ? 'Available' : 'Fallback Mode';

            if (hashMatch && dataMatch && seedMatch) {
                statusDiv.textContent = 'SELF-TEST PASSED ✓';
                statusDiv.className = 'test-status';
                statusDiv.style.color = 'var(--color-success)';
                detailsDiv.innerHTML = `
                    <strong>Test Results:</strong><br>
                    Test Size: 1 MB<br>
                    AI Model: ${aiStatus}<br>
                    Original Hash: ${originalHash.substring(0, 16)}...<br>
                    Restored Hash: ${restoredHash.substring(0, 16)}...<br>
                    Key Length: ${headerString.length} chars<br>
                    Partition: [${partition.join(', ')}]<br>
                    Positions Generated: ${positions.length}<br>
                    Data Match: ${dataMatch ? '✓' : '✗'}<br>
                    Hash Match: ${hashMatch ? '✓' : '✗'}<br>
                    Seed Recovery: ${seedMatch ? '✓' : '✗'}
                `;
                this.showSuccess('Self-test completed successfully!');
            } else {
                statusDiv.textContent = 'SELF-TEST FAILED ✗';
                statusDiv.className = 'test-status';
                statusDiv.style.color = 'var(--color-error)';
                detailsDiv.innerHTML = `
                    Data Match: ${dataMatch ? '✓' : '✗'}<br>
                    Hash Match: ${hashMatch ? '✓' : '✗'}<br>
                    Seed Recovery: ${seedMatch ? '✓' : '✗'}
                `;
                this.showError('Self-test failed - cryptographic operations may be compromised');
            }
        } catch (err) {
            statusDiv.textContent = 'SELF-TEST ERROR ✗';
            statusDiv.className = 'test-status';
            statusDiv.style.color = 'var(--color-error)';
            detailsDiv.textContent = `Error: ${err.message}`;
            this.showError(`Self-test error: ${err.message}`);
        }
    }

    downloadFile(data, filename) {
        if (!data) {
            this.showError('No data to download');
            return;
        }
        const blob = new Blob([data], { type: 'application/octet-stream' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024, sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}

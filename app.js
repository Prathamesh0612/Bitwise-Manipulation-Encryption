// NEC - Neural Encryption Corrector - AI-Enhanced Single-Key System
// Client-side implementation with Web Crypto API and TensorFlow.js
let necModel = null;

// Load NEC AI model
async function loadNECModel() {
    try {
        necModel = await tf.loadLayersModel('./'); // User's path
        console.log('✓ NEC AI model loaded successfully');
        updateAIStatus('✓ AI Model Active');
        return true;
    } catch (e) {
        console.log('⚠ NEC AI model not found, using heuristic fallback');
        necModel = null;
        updateAIStatus('100% Safe'); // User's status
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
    // This line updates the status in the header after loading
    const statusEl = document.getElementById('mode-status'); 
    if (statusEl) {
        statusEl.textContent = necModel ? 'AI Model Active' : 'Heuristic Mode Active';
        statusEl.style.color = necModel ? '#00cc99' : '#ff9900';
    }
    new NECApp();
});


class NECrypto {
    constructor() {
        // legacy defaults (some values may be replaced elsewhere)
        this.VERSION = 1;
        this.KDF_ITERATIONS = 150000;
        this.SALT_SIZE = 32;
        this.SEED_SIZE = 32;
        this.MAX_FILE_SIZE = 200 * 1024 * 1024;
        this.CHUNK_SIZE = 10 * 1024 * 1024;
        this.BASES = [12, 16, 20, 36];
        this.MIN_CORRUPTION_RATIO = 0.001;
        this.MAX_CORRUPTION_RATIO = 0.01;
        this.MAX_CRYPTO_RANDOM_BYTES = 65536;
    }

    async createCompactKey(keyString, salt, iterations, partition, bases, fileHashHex, seed, encodedStrings, originalBitLength) {
        // build header JSON and HMAC it
        const { masterKey, macKey } = await this.deriveKeysFromKeyString(keyString, salt, iterations);
        const encSeed = this.xorEncryptDecrypt(seed, masterKey);
        const headerObj = {
            version: this.VERSION,
            salt: btoa(String.fromCharCode(...salt)),
            iterations,
            encryptedSeed: btoa(String.fromCharCode(...encSeed)),
            partition,
            bases,
            fileHashHex,
            encodedStrings, // array of strings per partition
            originalBitLength // store the original bit count so we can strip padding on decrypt
        };
        const headerBytes = new TextEncoder().encode(JSON.stringify(headerObj));
        const tag = await this.computeHMAC(macKey, headerBytes);
        const combined = new Uint8Array(headerBytes.length + tag.length);
        combined.set(headerBytes, 0); combined.set(tag, headerBytes.length);
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
        if (combined.length < 32) throw new Error('Invalid header payload');
        const headerLen = combined.length - 32;
        const headerBytes = combined.slice(0, headerLen);
        const expectedTag = combined.slice(headerLen);

        // derive macKey to verify
        const hdrText = new TextDecoder().decode(headerBytes);
        const headerObj = JSON.parse(hdrText);
        try {
            const encodedInfo = Array.isArray(headerObj.encodedStrings) ? headerObj.encodedStrings.map(s => s ? s.length : 0) : [];
            console.log('parseCompactKeyFromHeader: header parsed', { version: headerObj.version, partition: headerObj.partition, bases: headerObj.bases, fileHashHex: headerObj.fileHashHex, encodedLengths: encodedInfo });
        } catch (e) {
            console.warn('parseCompactKeyFromHeader: header logging failed', e);
        }
        const salt = new Uint8Array(Array.from(atob(headerObj.salt)).map(c => c.charCodeAt(0)));
        const iterations = headerObj.iterations;

        const { masterKey, macKey } = await this.deriveKeysFromKeyString(keyString, salt, iterations);
        const ok = await this.verifyHMAC(macKey, headerBytes, expectedTag);
        if (!ok) throw new Error('Key verification failed - invalid key');

        const encSeed = new Uint8Array(Array.from(atob(headerObj.encryptedSeed)).map(c => c.charCodeAt(0)));
        const seed = this.xorEncryptDecrypt(encSeed, masterKey);

        return {
            version: headerObj.version,
            salt,
            iterations,
            partition: headerObj.partition,
            bases: headerObj.bases,
            fileHash: headerObj.fileHashHex,
            seed,
            encodedStrings: headerObj.encodedStrings,
            macKey,
            originalBitLength: headerObj.originalBitLength
        };
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

    
    // This is the single, correct analyzeFile function
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

        // --- HEURISTIC FALLBACK ---
        // We no longer generate ANY parameters here.
        // We only return the basic file analysis.
        const analysis = {
            size: bytes.length,
            entropy: this.calculateEntropy(bytes.slice(0, Math.min(1024 * 1024, bytes.length))),
            fileType: this.detectFileType(bytes),
            aiUsed: false
        };

        return analysis;
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

    // ------------------ Cryptographic & utility helpers ------------------
    generateRandomBytes(size) {
        // crypto.getRandomValues has a platform limit (commonly 65536 bytes).
        const maxChunk = this.MAX_CRYPTO_RANDOM_BYTES || 65536;
        if (size <= maxChunk) {
            const a = new Uint8Array(size);
            crypto.getRandomValues(a);
            return a;
        }
        const out = new Uint8Array(size);
        let offset = 0;
        while (offset < size) {
            const chunk = Math.min(maxChunk, size - offset);
            const tmp = new Uint8Array(chunk);
            crypto.getRandomValues(tmp);
            out.set(tmp, offset);
            offset += chunk;
        }
        return out;
    }

    generateKeyString() {
        // Human-friendly key string (base64 of 32 random bytes)
        const b = this.generateRandomBytes(32);
        let s = '';
        for (let i = 0; i < b.length; i++) s += String.fromCharCode(b[i]);
        return btoa(s);
    }

    bufferToHex(buf) {
        if (buf instanceof ArrayBuffer) buf = new Uint8Array(buf);
        return Array.from(buf).map(b => b.toString(16).padStart(2, '0')).join('');
    }

    hexToBuffer(hex) {
        const out = new Uint8Array(hex.length / 2);
        for (let i = 0; i < out.length; i++) out[i] = parseInt(hex.substr(i * 2, 2), 16);
        return out;
    }

    base85Encode(data) {
        let binary = '';
        for (let i = 0; i < data.length; i++) binary += String.fromCharCode(data[i]);
        return btoa(binary);
    }

    base85Decode(str) {
        const binary = atob(str);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
        return bytes;
    }

    async sha256(input) {
        let data;
        if (typeof input === 'string') data = new TextEncoder().encode(input);
        else data = input;
        const h = await crypto.subtle.digest('SHA-256', data);
        return this.bufferToHex(new Uint8Array(h));
    }

    async deriveKeysFromKeyString(keyString, salt, iterations) {
        const pw = new TextEncoder().encode(keyString);
        const baseKey = await crypto.subtle.importKey('raw', pw, { name: 'PBKDF2' }, false, ['deriveBits']);
        const bits = await crypto.subtle.deriveBits({ name: 'PBKDF2', salt: salt, iterations: iterations, hash: 'SHA-256' }, baseKey, 512);
        const buf = new Uint8Array(bits);
        const masterKey = buf.slice(0, 32);
        const macRaw = buf.slice(32, 64);
        const macKey = await crypto.subtle.importKey('raw', macRaw, { name: 'HMAC', hash: 'SHA-256' }, false, ['sign', 'verify']);
        return { masterKey, macKey };
    }

    async computeHMAC(macKey, data) {
        if (!(data instanceof Uint8Array)) data = new Uint8Array(data);
        const sig = await crypto.subtle.sign('HMAC', macKey, data);
        return new Uint8Array(sig);
    }

    async verifyHMAC(macKey, data, expected) {
        const sig = await this.computeHMAC(macKey, data);
        return this.constantTimeEquals(sig, expected);
    }

    xorEncryptDecrypt(data, key) {
        const out = new Uint8Array(data.length);
        for (let i = 0; i < data.length; i++) out[i] = data[i] ^ key[i % key.length];
        return out;
    }

    csprngInt(maxExclusive) {
        if (maxExclusive <= 0) return 0;
        const r = this.generateRandomBytes(4).reduce((a, b) => (a << 8) | b, 0) >>> 0;
        return r % maxExclusive;
    }

    generateRandomPartition(totalBits, parts, maxPart) {
        if (parts <= 0) return [];
        // split roughly evenly and distribute remainder
        const base = Math.floor(totalBits / parts);
        let rem = totalBits - base * parts;
        const out = new Array(parts).fill(base);
        for (let i = 0; i < parts && rem > 0; i++, rem--) out[i]++;
        return out;
    }

    constantTimeEquals(a, b) {
        if (a.length !== b.length) return false;
        let r = 0;
        for (let i = 0; i < a.length; i++) r |= a[i] ^ b[i];
        return r === 0;
    }

    async generateBitPositions(seed, fileHashHex, chunkId, partition, chunkBitLen) {
        // Deterministic PRF using repeated SHA-256 to generate pseudorandom integers
        const needed = partition.reduce((s, x) => s + x, 0);
        const positions = new Set();
        if (chunkBitLen <= 0 || needed <= 0) return [];
        let counter = 0;
        while (positions.size < needed) {
            const ctx = new Uint8Array(seed.length + fileHashHex.length + 8 + 4);
            ctx.set(seed, 0);
            ctx.set(new TextEncoder().encode(fileHashHex), seed.length);
            const dv = new DataView(ctx.buffer);
            dv.setUint32(seed.length + fileHashHex.length, chunkId || 0);
            dv.setUint32(seed.length + fileHashHex.length + 4, counter++);
            const hash = new Uint8Array(await crypto.subtle.digest('SHA-256', ctx));
            for (let i = 0; i + 3 < hash.length && positions.size < needed; i += 4) {
                const v = (hash[i] << 24) | (hash[i+1] << 16) | (hash[i+2] << 8) | hash[i+3];
                const pos = Math.abs(v) % chunkBitLen;
                positions.add(pos);
            }
        }
        // return flat array sorted
        return Array.from(positions).sort((a,b)=>a-b);
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

    // Bit helpers
    getBit(bytes, pos) {
        const byteIndex = Math.floor(pos / 8);
        const bitIndex = pos % 8;
        return (bytes[byteIndex] >> (7 - bitIndex)) & 1;
    }

    setBit(bytes, pos, val) {
        const byteIndex = Math.floor(pos / 8);
        const bitIndex = pos % 8;
        if (val) bytes[byteIndex] |= (1 << (7 - bitIndex));
        else bytes[byteIndex] &= ~(1 << (7 - bitIndex));
    }

    bytesToBits(bytes) {
        const bits = new Array(bytes.length * 8);
        for (let i = 0; i < bytes.length; i++) {
            for (let b = 0; b < 8; b++) bits[i * 8 + b] = (bytes[i] >> (7 - b)) & 1;
        }
        return bits;
    }

    bitsToBytes(bits) {
        const out = new Uint8Array(Math.ceil(bits.length / 8));
        out.fill(0);
        for (let i = 0; i < bits.length; i++) {
            if (bits[i]) {
                const byteIndex = Math.floor(i / 8);
                const bitIndex = i % 8;
                out[byteIndex] |= (1 << (7 - bitIndex));
            }
        }
        return out;
    }

    // BigInt <-> base-N
    bigintToBase(bi, base) {
        const alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';
        if (base < 2 || base > alphabet.length) throw new Error('Unsupported base');
        if (bi === 0n) return '0';
        let x = bi < 0n ? -bi : bi;
        let s = '';
        while (x > 0n) {
            const mod = x % BigInt(base);
            s = alphabet[Number(mod)] + s;
            x = x / BigInt(base);
        }
        return s;
    }

    baseToBigint(str, base) {
        const alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';
        if (base < 2 || base > alphabet.length) throw new Error('Unsupported base');
        let result = 0n;
        for (let i = 0; i < str.length; i++) {
            const idx = alphabet.indexOf(str[i]);
            if (idx < 0 || idx >= base) throw new Error('Invalid digit for base');
            result = result * BigInt(base) + BigInt(idx);
        }
        return result;
    }

    // Remove bits and encode per-partition into base-N strings
    async removeBitsAndEncode(bytes, positionsGroups, bases) {
        const bits = this.bytesToBits(bytes);
        const totalBits = bits.length;
        const removedMask = new Array(totalBits).fill(0);
        const encodedStrings = [];
        let outOfRangeCount = 0;
        for (let i = 0; i < positionsGroups.length; i++) {
            const posList = positionsGroups[i];
            const base = bases && bases[i] ? bases[i] : this.BASES[i % this.BASES.length];
            let bi = 0n;
            for (let j = 0; j < posList.length; j++) {
                const p = posList[j];
                if (p < 0 || p >= totalBits) {
                    console.error('removeBitsAndEncode: position out of range', { part: i, idx: j, pos: p, totalBits });
                    outOfRangeCount++;
                    continue;
                }
                const bit = bits[p];
                bi = (bi << 1n) | BigInt(bit);
                removedMask[p] = 1;
            }
            const s = this.bigintToBase(bi, base);
            encodedStrings.push(s);
        }
        const remainingBits = [];
        for (let i = 0; i < bits.length; i++) if (!removedMask[i]) remainingBits.push(bits[i]);
        const remainingBytes = this.bitsToBytes(remainingBits);
        try {
            const actualRemoved = positionsGroups.reduce((a,b)=>a+b.length,0) - outOfRangeCount;
            console.log('removeBitsAndEncode: totalBits, actualRemoved, skippedOutOfRange, remainingBitsLen, encodedLengths=', {
                totalBits,
                actualRemoved,
                outOfRangeCount,
                remainingBitsLen: remainingBits.length,
                check: 'totalBits(' + totalBits + ') - actualRemoved(' + actualRemoved + ') = ' + (totalBits - actualRemoved) + ' vs remainingBitsLen(' + remainingBits.length + ')'
            });
        } catch (e) { console.warn('removeBitsAndEncode: debug log failed', e); }
        return { remainingBytes, encodedStrings };
    }

    async decodeAndInsertBits(remainingBytes, positionsGroups, encodedStrings, bases) {
        const remainingBits = this.bytesToBits(remainingBytes);
        const totalBits = positionsGroups.reduce((a, b) => a + b.length, 0) + remainingBits.length;
        const reconstructed = new Array(totalBits).fill(null);
        for (let i = 0; i < positionsGroups.length; i++) {
            const posList = positionsGroups[i];
            const base = bases && bases[i] ? bases[i] : this.BASES[i % this.BASES.length];
            const s = encodedStrings[i] || '0';
            const bi = this.baseToBigint(s, base);
            const needed = posList.length;
            const partBits = new Array(needed).fill(0);
            // Extract bits in MSB-first order (matching the encoding direction)
            // The BigInt was built by shifting left and adding bits, so we extract
            // by counting down from the highest bit position.
            if (needed > 0) {
                // Find the highest set bit (or assume needed bits exist)
                let temp = bi;
                for (let k = 0; k < needed; k++) {
                    // Extract bits from MSB position down to LSB
                    const bitPos = needed - 1 - k;
                    partBits[k] = Number((temp >> BigInt(bitPos)) & 1n);
                }
            }
            for (let j = 0; j < posList.length; j++) {
                const p = posList[j];
                if (p < 0 || p >= reconstructed.length) {
                    console.error('decodeAndInsertBits: position out of range', { part: i, idx: j, pos: p, reconstructedLen: reconstructed.length });
                    continue;
                }
                reconstructed[p] = partBits[j];
            }
        }
        let remIdx = 0;
        for (let i = 0; i < reconstructed.length; i++) {
            if (reconstructed[i] === null) {
                reconstructed[i] = remainingBits[remIdx++] || 0;
            }
        }
        // check for nulls (shouldn't happen)
        const nullCount = reconstructed.reduce((c, v) => c + (v === null ? 1 : 0), 0);
        if (nullCount > 0) console.error('decodeAndInsertBits: reconstructed contains nulls', nullCount);
        const outBytes = this.bitsToBytes(reconstructed);
        try { console.log('decodeAndInsertBits: totalBits, outBytesLen, expectedBits=', { totalBits, outBytesLen: outBytes.length, expectedBits: reconstructed.length }); } catch(e){}
        return outBytes;
    }

    generateTestData(size = 1024 * 1024) { return this.generateRandomBytes(size); }

    // ==================== STREAMING / CHUNKED PROCESSING ====================
    // For large files on low-resource devices, process in chunks to avoid OOM

    async streamingRemoveBitsAndEncode(fileBuffer, positionsGroups, bases, yieldInterval = 50000) {
        // Optimized to avoid O(n) iterations on all bits
        // Instead: process only removed bit positions, then collect remaining bits by byte-range iteration
        const totalBits = fileBuffer.length * 8;
        const totalBytes = fileBuffer.length;
        const encodedStrings = [];

        // Step 1: Extract and encode removed bits per partition
        for (let partIdx = 0; partIdx < positionsGroups.length; partIdx++) {
            const posList = positionsGroups[partIdx];
            const base = bases[partIdx];
            let bi = 0n;

            for (let posIdx = 0; posIdx < posList.length; posIdx++) {
                const p = posList[posIdx];
                if (p < 0 || p >= totalBits) continue;
                const byteIndex = Math.floor(p / 8);
                const bitIndex = p % 8;
                const byte = fileBuffer[byteIndex];
                const bit = (byte >> (7 - bitIndex)) & 1;
                bi = (bi << 1n) | BigInt(bit);

                // Yield periodically to avoid blocking the UI
                if (posIdx % yieldInterval === 0) await new Promise(r => setTimeout(r, 0));
            }

            const s = this.bigintToBase(bi, base);
            encodedStrings.push(s);
        }

        // Step 2: Build a Set of removed positions for quick lookup
        const removedPositions = new Set();
        for (const posList of positionsGroups) {
            for (const p of posList) removedPositions.add(p);
        }

        // Step 3: Collect remaining bits by iterating bytes and bits within each byte
        // This is O(n) but only in the byte-level outer loop, not the full bit iteration
        const remainingBitsArray = [];
        let bitCount = 0;
        for (let byteIdx = 0; byteIdx < totalBytes; byteIdx++) {
            const byte = fileBuffer[byteIdx];
            for (let bitIdx = 0; bitIdx < 8; bitIdx++) {
                const globalBitPos = byteIdx * 8 + bitIdx;
                if (globalBitPos >= totalBits) break; // safety check for last partial byte
                if (!removedPositions.has(globalBitPos)) {
                    const bit = (byte >> (7 - bitIdx)) & 1;
                    remainingBitsArray.push(bit);
                }
            }
            // Yield periodically to prevent UI freeze
            if (byteIdx % (yieldInterval * 8) === 0) {
                await new Promise(r => setTimeout(r, 0));
            }
        }

        const remainingBytes = this.bitsToBytes(remainingBitsArray);
        console.log('streamingRemoveBitsAndEncode: totalBits, removedCount, remainingBytesLen, encodedLengths=', {
            totalBits,
            removedCount: positionsGroups.reduce((a, b) => a + b.length, 0),
            remainingBytesLen: remainingBytes.length,
            encodedLengths: encodedStrings.map(s => s.length)
        });
        return { remainingBytes, encodedStrings };
    }

    async streamingDecodeAndInsertBits(remainingBytes, positionsGroups, encodedStrings, bases) {
        // Reconstruct the original file by inserting bits back at their positions
        // This also works in-place without requiring full file in memory during bit insertion
        const remainingBits = this.bytesToBits(remainingBytes);
        const totalBits = positionsGroups.reduce((a, b) => a + b.length, 0) + remainingBits.length;

        // Build a map of position -> bit value for quick lookup
        const positionToBit = new Map();
        for (let partIdx = 0; partIdx < positionsGroups.length; partIdx++) {
            const posList = positionsGroups[partIdx];
            const base = bases[partIdx];
            const s = encodedStrings[partIdx] || '0';
            const bi = this.baseToBigint(s, base);
            const needed = posList.length;

            // Extract bits in MSB-first order
            if (needed > 0) {
                for (let k = 0; k < needed; k++) {
                    const bitPos = needed - 1 - k;
                    const bit = Number((bi >> BigInt(bitPos)) & 1n);
                    const p = posList[k];
                    positionToBit.set(p, bit);
                }
            }
        }

        // Reconstruct the bit array by filling positions with their bits, and remaining slots with remaining bits
        const reconstructed = new Array(totalBits).fill(null);
        for (const [pos, bit] of positionToBit.entries()) {
            if (pos >= 0 && pos < totalBits) {
                reconstructed[pos] = bit;
            }
        }

        let remIdx = 0;
        for (let i = 0; i < reconstructed.length; i++) {
            if (reconstructed[i] === null) {
                reconstructed[i] = remainingBits[remIdx++] || 0;
            }
        }

        const outBytes = this.bitsToBytes(reconstructed);
        console.log('streamingDecodeAndInsertBits: totalBits, outBytesLen=', { totalBits, outBytesLen: outBytes.length });
        return outBytes;
    }
}




// -----------------------------------------------------------------
// CLASS 2: NECApp (The UI controller)
// -----------------------------------------------------------------
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
            copyKeyBtn.addEventListener('click', (e) => {
                 // Use currentTarget to ensure we get the button that was clicked
                 const targetId = e.currentTarget.dataset.target;
                 if(targetId) this.copyToClipboard(targetId);
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
        
        // --- THIS IS THE CRITICAL LINE THAT MAKES THE BUTTON WORK ---
        document.getElementById('start-encrypt').addEventListener('click', () => this.encryptFile());
        
        document.getElementById('start-decrypt').addEventListener('click', () => this.decryptFile());
        document.getElementById('run-self-test').addEventListener('click', () => this.runSelfTest());
                
        document.getElementById('download-encrypted').addEventListener('click', () =>
            this.downloadFile(this.encryptedData, `${this.currentFile?.name || 'encrypted'}.nec`) // Fixed .bme to .nec
        );
        document.getElementById('download-restored').addEventListener('click', () =>
            this.downloadFile(this.restoredData, this.originalFileName || 'restored.bin')
        );
        this.createDemoFileButton();
    }

    createDemoFileButton() {
        const uploadArea = document.getElementById('encrypt-upload');
        if (!uploadArea) return; // Add check in case element doesn't exist
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
        if (!area || !input) return; // Add checks

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

        let ratioText, partText, aiStatus;

        if (analysis.aiUsed) {
            // If AI is active, we know the parameters
            ratioText = `${(analysis.corruptionRatio * 100).toFixed(3)}%`;
            partText = analysis.partitionCount;
            aiStatus = '✓ AI Enhanced';
        } else {
            // If AI is NOT active (heuristic), parameters will be
            // generated at the moment of encryption.
            ratioText = 'Dynamic (on encrypt)';
            partText = 'Dynamic (on encrypt)';
            aiStatus = '⚠ Heuristic Mode';
        }

        document.getElementById('encrypt-ratio').textContent = ratioText;
        document.getElementById('encrypt-partitions').textContent = partText;
        updateAIStatus(aiStatus);

        document.getElementById('encrypt-info').classList.remove('hidden');
        this.fileAnalysis = analysis;
    }

    // --- THIS IS THE encryptFile FUNCTION IN THE CORRECT CLASS ---
    async encryptFile() {
        if (!this.currentFile) {
             this.showError('No file selected for encryption.');
             return;
        }
        const t0 = Date.now();
        this.showProgress('encrypt', 0);
        
        try {
            // --- DYNAMIC PARAMETER GENERATION ---
            let corruptionRatio, partitionCount, bases;

            if (this.fileAnalysis.aiUsed) {
                // 1. Get parameters from the AI analysis
                corruptionRatio = this.fileAnalysis.corruptionRatio;
                partitionCount = this.fileAnalysis.partitionCount;
                // Generate bases matching partitionCount (need one base per partition)
                const shuffledBases = [...this.crypto.BASES].sort(() => 0.5 - Math.random());
                bases = [];
                for (let i = 0; i < partitionCount; i++) {
                    bases.push(shuffledBases[i % shuffledBases.length]);
                }
                
            } else {
                // 2. HEURISTIC: Generate NEW random parameters *every time*
                
                // Get a crypto-safe random float [0, 1]
                const randomVal = (this.crypto.generateRandomBytes(4).reduce((a, b) => a * 256 + b, 0)) / 0xFFFFFFFF;
                
                // Generate a random corruption ratio
                corruptionRatio = this.crypto.MIN_CORRUPTION_RATIO + (randomVal * (this.crypto.MAX_CORRUPTION_RATIO - this.crypto.MIN_CORRUPTION_RATIO));

                // Generate a random partition count (2 to 8)
                partitionCount = 2 + this.crypto.csprngInt(7); 

                // Generate bases matching partitionCount (need one base per partition)
                const shuffledBases = [...this.crypto.BASES].sort(() => 0.5 - Math.random());
                // Use as many bases as we have, but at least partitionCount (cycle through if needed)
                bases = [];
                for (let i = 0; i < partitionCount; i++) {
                    bases.push(shuffledBases[i % shuffledBases.length]);
                }
                
                // --- UPDATE UI to show what we generated ---
                document.getElementById('encrypt-ratio').textContent = `${(corruptionRatio * 100).toFixed(3)}%`;
                document.getElementById('encrypt-partitions').textContent = partitionCount;
            }
            // --- END DYNAMIC PARAMETER GENERATION ---


            const salt = this.crypto.generateRandomBytes(32);
            const seed = this.crypto.generateRandomBytes(32);
            const keyString = this.crypto.generateKeyString();

            const data = new Uint8Array(await this.currentFile.arrayBuffer());
            const fileHash = await this.crypto.sha256(data);
            document.getElementById('original-hash').textContent = fileHash;

            const totalBits = data.length * 8;
            
            // --- USE THE NEW DYNAMIC VARIABLES ---
            const partition = this.crypto.generateRandomPartition(
                Math.floor(totalBits * corruptionRatio), // Use dynamic ratio
                partitionCount,                          // Use dynamic count
                partitionCount
            );

            this.showProgress('encrypt', 25);

            // Removal-based scheme: generate positions grouped per partition across whole file
            // For each partition, generate positions within the whole file bit range. We'll produce an array of arrays.
            const positionsGroups = [];
            // generate per-partition bit positions deterministically (we call generateBitPositions per partition index)
            for (let pIdx = 0; pIdx < partition.length; pIdx++) {
                // generate as if chunkId==pIdx to keep deterministic but unique context
                const posList = await this.crypto.generateBitPositions(seed, fileHash, pIdx, [partition[pIdx]], totalBits);
                positionsGroups.push(posList.slice(0, partition[pIdx]));
            }

            try {
                console.log('encrypt: partition count=' + partition.length + ' bases count=' + bases.length + ' bases=' + JSON.stringify(bases));
                console.log('encrypt: partition, bases, positionsCounts, firstPositions (per part):', {
                    partition, bases,
                    positionsCounts: positionsGroups.map(g => g.length),
                    positionsPreview: positionsGroups.map(g => g.slice(0,5))
                });
            } catch (e) { console.warn('encrypt: debug log failed', e); }

            // Use streaming for large files (>10MB) to reduce memory usage
            let remainingBytes, encodedStrings;
            if (data.length > 10 * 1024 * 1024) {
                console.log('encrypt: using streaming mode for large file');
                const result = await this.crypto.streamingRemoveBitsAndEncode(data, positionsGroups, bases);
                remainingBytes = result.remainingBytes;
                encodedStrings = result.encodedStrings;
            } else {
                const result = await this.crypto.removeBitsAndEncode(data, positionsGroups, bases);
                remainingBytes = result.remainingBytes;
                encodedStrings = result.encodedStrings;
            }

            try {
                console.log('encrypt: encodedStrings lengths', encodedStrings.map(s => s.length));
            } catch (e) { console.warn('encrypt: encodedStrings log failed', e); }

            const encryptedData = remainingBytes;

            this.showProgress('encrypt', 75);
            const encryptedHash = await this.crypto.sha256(encryptedData);
            document.getElementById('encrypted-hash').textContent = encryptedHash;

            // --- USE THE NEW DYNAMIC 'bases' VARIABLE ---
            // Pass the original bit length so we can strip padding during decryption
            const originalBitLength = data.length * 8;
            const headerString = await this.crypto.createCompactKey(
                keyString, salt, this.crypto.KDF_ITERATIONS, partition, bases, fileHash, seed, encodedStrings, originalBitLength
            );

            console.log('encrypt: headerString length', headerString.length, 'preview:', headerString.slice(0, 128));
            
            // Debug: show what's actually in the header
            try {
                const checksum = headerString.substring(3, 11);
                const base = headerString.substring(11);
                const combined = this.crypto.base85Decode(base);
                const headerLen = combined.length - 32;
                const headerBytes = combined.slice(0, headerLen);
                const hdrText = new TextDecoder().decode(headerBytes);
                const headerObj = JSON.parse(hdrText);
                console.log('encrypt: header contents - partition.length=' + headerObj.partition.length + ' bases.length=' + (headerObj.bases || []).length + ' bases=' + JSON.stringify(headerObj.bases));
            } catch (e) { console.warn('encrypt: header inspection failed', e); }

            const preface = `NECHDR\n${headerString}\nENDHDR\n`;
            const prefaceBytes = new TextEncoder().encode(preface);
            const finalData = new Uint8Array(prefaceBytes.length + encryptedData.length);
            finalData.set(prefaceBytes, 0);
            finalData.set(encryptedData, prefaceBytes.length);
            this.encryptedData = finalData;
            
            // Debug: show exact byte positions
            const startMarkerBytes = new TextEncoder().encode('NECHDR\n');
            const endMarkerBytes = new TextEncoder().encode('\nENDHDR\n');
            console.log('encrypt: marker lengths:', {
                startMarkerLen: startMarkerBytes.length,
                endMarkerLen: endMarkerBytes.length,
                headerStringLen: headerString.length,
                headerStringBytes: new TextEncoder().encode(headerString).length
            });
            
            console.log('encrypt: file layout:', {
                prefaceLen: prefaceBytes.length,
                encryptedDataLen: encryptedData.length,
                totalLen: finalData.length,
                expectedDataStart: startMarkerBytes.length + new TextEncoder().encode(headerString).length + endMarkerBytes.length
            });

            this.showProgress('encrypt', 100);

            document.getElementById('encryption-key').value = keyString;
            document.getElementById('encrypt-results').classList.remove('hidden');

            const t1 = Date.now();
            const thr = (data.length / ((t1 - t0) || 1) / 1000) / (1024 * 1024); // Avoid divide by zero
            document.getElementById('encrypt-throughput').textContent = `${thr.toFixed(1)} MB/s`;
            
            const aiNote = this.fileAnalysis.aiUsed ? ' (AI-Enhanced)' : ' (Dynamic Heuristic)';
            this.showSuccess(`File encrypted successfully${aiNote}! Save your key and the .nec file.`);
        } catch (err) {
            this.showError(`Encryption failed: ${err.message}`);
            // Hide progress bar on failure
            document.getElementById('encrypt-progress').classList.add('hidden');
        }
    }
    // --- END OF encryptFile FUNCTION ---

    async decryptFile() {
        if (!this.encryptedFile) {
            this.showError('No file selected for decryption.');
            return;
        }
        const keyString = document.getElementById('decrypt-key').value.trim();
        if (!keyString) {
            this.showError('Please provide the encryption key');
            return;
        }

        const t0 = Date.now();
        this.showProgress('decrypt', 0);
        
        try {
            let raw;
            try {
                raw = new Uint8Array(await this.encryptedFile.arrayBuffer());
            } catch (readErr) {
                console.error('decrypt: failed to read encryptedFile.arrayBuffer()', readErr);
                this.showError('Failed to read the selected file. Please re-select the .nec file (it may have been moved, deleted, or permissioned).');
                // Focus and clear the file input to prompt user to reselect
                const input = document.getElementById('decrypt-file');
                if (input) {
                    try { input.value = null; input.click(); } catch (e) { /* ignore */ }
                }
                // Stop decryption flow
                return;
            }

            // Byte-level search for header markers (more robust than string-based decoding)
            const startMarker = 'NECHDR\n';
            const endMarker = '\nENDHDR\n';
            const startMarkerBytes = new TextEncoder().encode(startMarker);
            const endMarkerBytes = new TextEncoder().encode(endMarker);

            // Helper to find subarray (like indexOf for Uint8Array)
            const indexOfSubarray = (buf, pat, from = 0) => {
                const limit = buf.length - pat.length;
                outer: for (let i = from; i <= limit; i++) {
                    for (let j = 0; j < pat.length; j++) if (buf[i + j] !== pat[j]) continue outer;
                    return i;
                }
                return -1;
            };

            // Search for markers starting from the beginning
            let startIdx = indexOfSubarray(raw, startMarkerBytes, 0);
            if (startIdx < 0) throw new Error('Missing key header in file');

            // Search for end marker starting from after the start marker
            let endIdx = indexOfSubarray(raw, endMarkerBytes, startIdx + startMarkerBytes.length);
            if (endIdx < 0) {
                // Try expanding search in case of large files; scan more aggressively
                const EXPANDED_SCAN = Math.min(raw.length, 5 * 1024 * 1024); // 5 MB
                endIdx = indexOfSubarray(raw.subarray(0, EXPANDED_SCAN), endMarkerBytes, startIdx + startMarkerBytes.length);
                if (endIdx >= 0) {
                    endIdx = endIdx; // Already correct since we're searching from 0
                } else {
                    endIdx = -1;
                }
            }
            if (endIdx < 0) throw new Error('Corrupted key header in file (END marker missing)');

            console.log('Header scan (bytes):', { startIdx, endIdx, startMarkerLen: startMarkerBytes.length, endMarkerLen: endMarkerBytes.length });

            // Extract header bytes and decode safely
            // startIdx = byte index of 'N' in 'NECHDR\n'
            // endIdx = byte index of '\n' in '\nENDHDR\n' (the starting \n)
            // headerContent should be from after 'NECHDR\n' to before '\nENDHDR\n'
            const headerBytes = raw.slice(startIdx + startMarkerBytes.length, endIdx);
            const headerString = new TextDecoder().decode(headerBytes);
            // remainingBytes start after the complete '\nENDHDR\n' marker
            const headerBytesLen = endIdx + endMarkerBytes.length;
            
            console.log('decrypt: header parsing details:', { 
                startIdx, 
                endIdx, 
                startMarkerLen: startMarkerBytes.length,
                endMarkerLen: endMarkerBytes.length,
                headerContentLen: headerBytes.length,
                headerStringLen: headerString.length,
                computedHeaderBytesLen: headerBytesLen,
                rawLength: raw.length,
                firstMarkerBytes: Array.from(raw.slice(startIdx, startIdx + startMarkerBytes.length)),
                lastMarkerBytes: Array.from(raw.slice(endIdx, endIdx + endMarkerBytes.length))
            });

            const keyData = await this.crypto.parseCompactKeyFromHeader(headerString, keyString);
            this.showProgress('decrypt', 25);

            const remainingBytes = raw.slice(headerBytesLen);
            try {
                console.log('decrypt: headerBytesLen=', headerBytesLen, 'remainingBytesLen=', remainingBytes.length);
                console.log('decrypt: parsed header partition/bases/encodedLengths=', {
                    partition: keyData.partition,
                    bases: keyData.bases,
                    encodedLengths: (keyData.encodedStrings || []).map(s => s ? s.length : 0)
                });
                console.log('decrypt: loaded from header - originalBitLength=', keyData.originalBitLength, 'expectedOriginalFileHash=', keyData.fileHash);
            } catch (e) { console.warn('decrypt: debug logs failed', e); }

            // Recompute positionsGroups deterministically (same method used at encryption)
            const positionsGroups = [];
            // Use the original bit length stored in the header to compute positions correctly
            const totalBits = (keyData.originalBitLength && Number.isFinite(keyData.originalBitLength)) ? keyData.originalBitLength: (remainingBytes.length * 8 + keyData.partition.reduce((a, b) => a + b, 0));
            for (let pIdx = 0; pIdx < keyData.partition.length; pIdx++) {
                const posList = await this.crypto.generateBitPositions(keyData.seed, keyData.fileHash, pIdx, [keyData.partition[pIdx]], totalBits);
                positionsGroups.push(posList.slice(0, keyData.partition[pIdx]));
                if (pIdx < 4) console.log(`decrypt: positions preview for part ${pIdx}:`, positionsGroups[pIdx].slice(0,8));
            }

            // Use streaming for large files to reduce memory usage
            let reconstructed;
            if (remainingBytes.length > 10 * 1024 * 1024) {
                console.log('decrypt: using streaming mode for large file');
                reconstructed = await this.crypto.streamingDecodeAndInsertBits(remainingBytes, positionsGroups, keyData.encodedStrings || [], keyData.bases);
            } else {
                reconstructed = await this.crypto.decodeAndInsertBits(remainingBytes, positionsGroups, keyData.encodedStrings || [], keyData.bases);
            }
            
            // Strip padding bits if originalBitLength is stored in header
            try {
                const bitsBeforePadding = reconstructed.length * 8;
                const removedCount = keyData.partition.reduce((a,b)=>a+b,0);
                const remainingBitsCount = remainingBytes.length * 8;
                console.log('decrypt: reconstruction - bitsBeforePadding=' + bitsBeforePadding + ' removedBits=' + removedCount + ' remainingBits=' + remainingBitsCount + ' sum=' + (removedCount + remainingBitsCount));
            } catch (e) { console.warn('decrypt: reconstruction log failed', e); }
            
            if (keyData.originalBitLength && keyData.originalBitLength < reconstructed.length * 8) {
                const paddingBits = (reconstructed.length * 8) - keyData.originalBitLength;
                console.log('decrypt: stripping', paddingBits, 'padding bits, reconstructed was', reconstructed.length, 'bytes, will be', Math.ceil(keyData.originalBitLength/8), 'bytes');
                const bits = this.crypto.bytesToBits(reconstructed);
                const originalBits = bits.slice(0, keyData.originalBitLength);
                reconstructed = this.crypto.bitsToBytes(originalBits);
            }
            
            const restored = reconstructed;

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
            this.originalFileName = this.encryptedFile.name.replace(/\.nec$/i, '') || 'restored.bin'; // Fixed .bme to .nec


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
            const thr = (restored.length / ((t1 - t0) || 1) / 1000) / (1024 * 1024); // Avoid divide by zero
            document.getElementById('decrypt-throughput').textContent = `${thr.toFixed(1)} MB/s`;
        } catch (err) {
            this.showError(`Decryption failed: ${err.message}`);
            // Hide progress bar on failure
            document.getElementById('decrypt-progress').classList.add('hidden');
        }
    }

    async runSelfTest() {
        const resultsDiv = document.getElementById('self-test-results');
        const statusDiv = resultsDiv.querySelector('.test-status');
        const detailsDiv = resultsDiv.querySelector('.test-details');
        if (!resultsDiv || !statusDiv || !detailsDiv) return; // Add checks
        
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

            // generate positionsGroups and remove bits
            const positionsGroups = [];
            const totalBits = testData.length * 8;
            for (let pIdx = 0; pIdx < partition.length; pIdx++) {
                const posList = await this.crypto.generateBitPositions(seed, originalHash, pIdx, [partition[pIdx]], totalBits);
                positionsGroups.push(posList.slice(0, partition[pIdx]));
            }

            const { remainingBytes, encodedStrings } = await this.crypto.removeBitsAndEncode(testData, positionsGroups, bases);

            const originalBitLength = testData.length * 8;
            const headerString = await this.crypto.createCompactKey(
                keyString, salt, this.crypto.KDF_ITERATIONS, partition, bases, originalHash, seed, encodedStrings, originalBitLength
            );
            const keyData = await this.crypto.parseCompactKeyFromHeader(headerString, keyString);

            // Recompute positionsGroups and restore
            const positionsGroups2 = [];
            for (let pIdx = 0; pIdx < keyData.partition.length; pIdx++) {
                const posList = await this.crypto.generateBitPositions(keyData.seed, keyData.fileHash, pIdx, [keyData.partition[pIdx]], totalBits);
                positionsGroups2.push(posList.slice(0, keyData.partition[pIdx]));
            }
            let restoredData = await this.crypto.decodeAndInsertBits(remainingBytes, positionsGroups2, keyData.encodedStrings || [], keyData.bases);
            
            // Log reconstruction details
            try {
                const removedCount = keyData.partition.reduce((a,b)=>a+b,0);
                const remainingBitsCount = remainingBytes.length * 8;
                const bitsBeforePadding = restoredData.length * 8;
                console.log('selfTest: reconstruction - bitsBeforePadding=' + bitsBeforePadding + ' removedBits=' + removedCount + ' remainingBits=' + remainingBitsCount + ' sum=' + (removedCount + remainingBitsCount));
            } catch (e) { console.warn('selfTest: reconstruction log failed', e); }
            
            // Strip padding bits if stored in header
            if (keyData.originalBitLength && keyData.originalBitLength < restoredData.length * 8) {
                const paddingBits = (restoredData.length * 8) - keyData.originalBitLength;
                const bits = this.crypto.bytesToBits(restoredData);
                const originalBits = bits.slice(0, keyData.originalBitLength);
                restoredData = this.crypto.bitsToBytes(originalBits);
            }

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
    
    // --- UTILITY FUNCTIONS ---
    
    showProgress(type, percent) {
        const fill = document.getElementById(`${type}-progress-fill`);
        const text = document.getElementById(`${type}-progress-text`);
        const bar = document.getElementById(`${type}-progress`);
        if (!fill || !text || !bar) return; // Add checks
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
        if (active) {
            active.insertBefore(div, active.firstChild);
        }
        setTimeout(() => { if (div.parentNode) div.parentNode.removeChild(div); }, 5000);
    }
    
    async copyToClipboard(elementId) {
        const el = document.getElementById(elementId);
        const btn = document.querySelector(`[data-target="${elementId}"]`);
        if (!el) {
            console.error('No element to copy from with ID:', elementId);
            return;
        }
        if (!btn) {
            console.error('No copy button found for target:', elementId);
            return;
        }
        
        try {
            await navigator.clipboard.writeText(el.value);
            const orig = btn.innerHTML; // Store the icon + text
            btn.textContent = 'Copied!';
            setTimeout(() => btn.innerHTML = orig, 2000);
        } catch {
            el.select();
            document.execCommand('copy');
            const orig = btn.innerHTML;
            btn.textContent = 'Copied!';
            setTimeout(() => btn.innerHTML = orig, 2000);
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
# SHAPECRYPT
> **Shape-Based Pixel Shuffling with AES-256-GCM for Secure Image Encryption**

A dual-layer, college-level image cryptography system and web application combining deterministic geometric pixel permutation (Layer 1) with authenticated AES-256-GCM symmetric encryption (Layer 2).

---

## 1. System Architecture & Security Model

SHAPECRYPT is intentionally structured with a rigorous separation between visual/spatial data transformation and modern cryptographic security:

```
+-------------------------------------------------------------------------------+
|                                INPUT PLAIN IMAGE                              |
+-------------------------------------------------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
| LAYER 1: CUSTOM GEOMETRIC PIXEL PERMUTATION & RGB SHUFFLE                     |
|                                                                               |
|  * Shape Coordinate Generation:                                               |
|      Circle / Triangle / Square / Star / Spiral                               |
|  * Deterministic traversal of pixel coordinates driven by key/seed            |
|  * Exact mathematical inverse mapping guarantees lossless recovery            |
|  * Multi-pattern chaining support: e.g. Circle -> Triangle -> Spiral         |
|  * Key-dependent RGB channel permutation (RGB, RBG, GRB, GBR, BRG, BGR)       |
|                                                                               |
|  NOTE: Geometric pixel shuffling provides visual obfuscation and breaks       |
|        spatial autocorrelation; it does NOT replace cryptographic security.   |
+-------------------------------------------------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
| LAYER 2: AES-256-GCM CRYPTOGRAPHIC SECURITY LAYER                             |
|                                                                               |
|  * Key Derivation: Password + CSPRNG Salt -> Argon2id / PBKDF2 (256-bit key)  |
|  * Authenticated Encryption: AES-256-GCM with 96-bit CSPRNG Nonce/IV          |
|  * Integrity & Authenticity: 128-bit authentication tag verification          |
|  * Zero plaintext password retention: Password is never stored or exposed     |
|  * Cryptographic security guarantee: IND-CPA and IND-CCA2 security            |
+-------------------------------------------------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                          ENCRYPTED PACKAGE / CIPHERTEXT                       |
+-------------------------------------------------------------------------------+
```

### Decryption Pipeline (Exact Reversal)
1. Read encrypted package and extract non-secret metadata (Salt, Nonce, Shape Order, Image Dimensions).
2. Derive 256-bit AES key from user-provided password and stored salt.
3. Authenticate and decrypt ciphertext using AES-256-GCM (fails immediately if password is wrong or ciphertext is tampered with).
4. Deserialize decrypted bytes into the shuffled pixel array.
5. Invert the key-dependent RGB channel permutation.
6. Apply inverse geometric coordinate permutations in reverse sequence.
7. Reconstruct the exact original image matrix (`Decrypt(Encrypt(image)) == image`).

---

## 2. Directory Structure

```
SHAPECRYPT/
│
├── app.py                  # Flask web application entrypoint & API routes
├── requirements.txt        # Python package dependencies
├── README.md               # Project documentation and architectural overview
├── .gitignore              # Git ignore rules for caches, envs, and temporary data
│
├── encryption/             # Layer 1 & Layer 2 encryption implementation
│   ├── __init__.py         # Package initialization
│   ├── shapes.py           # Geometric coordinate traversal generators (Circle, Triangle, Square, Star, Spiral)
│   ├── permutation.py      # Deterministic pixel permutation engine & coordinate mapping
│   ├── rgb_shuffle.py      # Key-dependent RGB channel permutation
│   ├── aes.py              # Key derivation (Argon2id/PBKDF2) & AES-256-GCM encryption
│   └── pipeline.py         # Full encryption pipeline orchestrator
│
├── decryption/             # Decryption & inverse transformation implementation
│   ├── __init__.py         # Package initialization
│   └── pipeline.py         # Full authenticated decryption and inverse permutation orchestrator
│
├── analysis/               # Cryptographic & statistical evaluation metrics
│   ├── __init__.py         # Package initialization
│   ├── entropy.py          # Shannon entropy calculation (ideal: ~8.0000 bits/symbol)
│   ├── histogram.py        # RGB intensity histogram distribution analysis
│   ├── correlation.py      # Adjacent pixel correlation (Horizontal, Vertical, Diagonal)
│   └── metrics.py          # Differential cryptanalysis metrics: NPCR and UACI
│
├── templates/              # HTML frontend templates
│   └── index.html          # Web interface for upload, pattern selection, encryption, & analysis
│
├── static/                 # Static web assets
│   ├── css/
│   │   └── style.css       # Modern responsive stylesheet
│   └── js/
│       └── app.js          # Client-side interactions and API communication
│
├── uploads/                # Temporary holding directory for uploaded source images
└── output/                 # Temporary holding directory for encrypted packages and outputs
```

---

## 3. Directory Purpose Reference

| Directory / File | Role & Purpose |
|---|---|
| `app.py` | Flask backend service providing endpoints for image upload, encryption, decryption, and metrics. |
| `encryption/` | Encapsulates all forward transformation algorithms: geometric shape traversal, pixel coordinate swapping, RGB reordering, and authenticated AES-GCM encryption. |
| `decryption/` | Encapsulates reverse operations: GCM authentication check, decryption, inverse RGB reordering, and inverse coordinate traversal to restore the exact original image. |
| `analysis/` | Provides rigorous academic metrics to measure encryption quality: Shannon entropy, adjacent pixel correlation coefficients, histogram uniformity, NPCR, and UACI. |
| `templates/` & `static/` | Clean, interactive web UI allowing users to upload images, choose single or chained shapes, enter passwords, decrypt packages, and view side-by-side comparisons and charts. |
| `uploads/` & `output/` | Sandboxed directories on the server for temporary file handling during user operations. |

---

## 4. Setup & Running on Windows

### Prerequisites
- Python 3.10+ (Python 3.14 recommended)
- `pip` package manager

### Step 1: Install Dependencies
Open PowerShell or Command Prompt in the `SHAPECRYPT` project directory:
```powershell
python -m pip install -r requirements.txt
```

### Step 2: Run the Flask Application
```powershell
python app.py
```

### Step 3: Access the Application
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 5. Development Roadmap

- [x] **Phase 1: Project Foundation** (Folder hierarchy, package stubs, environment validation, minimal Flask app, web UI skeleton)
- [ ] **Phase 2: Layer 1 Geometric Pixel Permutations** (Circle, Triangle, Square, Star, Spiral coordinate generators and reversible permutation mapping)
- [ ] **Phase 3: Layer 1.5 & Layer 2 Security Engine** (RGB channel permutation, Argon2id/PBKDF2 key derivation, AES-256-GCM encryption/decryption)
- [ ] **Phase 4: Encryption & Decryption Pipelines** (Multi-pattern chaining, serialization, exact lossless reconstruction tests)
- [ ] **Phase 5: Academic Security Analysis Suite** (Shannon entropy, histogram comparisons, adjacent pixel correlation, NPCR, UACI)
- [ ] **Phase 6: Full Web Application Integration & UI Polish** (Image drag-and-drop, real-time metrics charts, side-by-side image diffs)

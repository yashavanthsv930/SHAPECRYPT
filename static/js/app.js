// app.js - handle encrypt/decrypt form submissions via fetch API
document.addEventListener('DOMContentLoaded', () => {
  const statusDot = document.getElementById('status-dot');
  const statusLabel = document.getElementById('status-label');

  // Helper to update backend status indicator
  const setStatus = (text, color) => {
    statusLabel.textContent = text;
    statusDot.style.backgroundColor = color;
  };

  // Initial health check
  fetch('/')
    .then(res => {
      if (res.ok) setStatus('Backend OK', '#28a745');
      else setStatus('Backend Unavailable', '#dc3545');
    })
    .catch(() => setStatus('Backend Unavailable', '#dc3545'));

  const encryptForm = document.getElementById('encrypt-form');
  const decryptForm = document.getElementById('decrypt-form');

  encryptForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    setStatus('Encrypting...', '#ffc107');
    const formData = new FormData(encryptForm);
    // patterns field may be comma-separated string; split into list
    const patternsVal = formData.get('patterns');
    if (patternsVal) {
      // Convert to list by splitting on commas and trimming
      const list = patternsVal.split(',').map(p => p.trim()).filter(p => p);
      // remove original and set multiple values
      formData.delete('patterns');
      list.forEach(p => formData.append('patterns', p));
    }
    try {
      const resp = await fetch('/encrypt', {
        method: 'POST',
        body: formData,
      });
      if (!resp.ok) {
        const detail = await resp.text();
        let message = detail;
        try { message = JSON.parse(detail).error || detail; } catch (_) {}
        throw new Error(message || `Encryption failed (HTTP ${resp.status})`);
      }
      const blob = await resp.blob();
      const filename = resp.headers.get('Content-Disposition')?.match(/filename="?([^";]+)"?/)?.[1] || 'encrypted.shapecrypt';
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
      setStatus('Encryption succeeded', '#28a745');
    } catch (err) {
      console.error('SHAPECRYPT encryption error:', err);
      setStatus(`Encryption error: ${err.message}`, '#dc3545');
    }
  });

  decryptForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    setStatus('Decrypting...', '#ffc107');
    const formData = new FormData(decryptForm);
    try {
      const resp = await fetch('/decrypt', {
        method: 'POST',
        body: formData,
      });
      if (!resp.ok) {
        const detail = await resp.text();
        let message = detail;
        try { message = JSON.parse(detail).error || detail; } catch (_) {}
        throw new Error(message || `Decryption failed (HTTP ${resp.status})`);
      }
      const blob = await resp.blob();
      const filename = resp.headers.get('Content-Disposition')?.match(/filename="?([^";]+)"?/)?.[1] || 'decrypted.png';
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      // Delay revoking to ensure download starts
      setTimeout(() => {
        window.URL.revokeObjectURL(url);
      }, 1000);
      setStatus('Decryption succeeded', '#28a745');
    } catch (err) {
      console.error('SHAPECRYPT decryption error:', err);
      setStatus(`Decryption error: ${err.message}`, '#dc3545');
    }
  });
});

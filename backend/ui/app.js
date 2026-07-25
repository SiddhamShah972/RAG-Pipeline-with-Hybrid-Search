document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const chatForm = document.getElementById('chat-form');
    const queryInput = document.getElementById('query-input');
    const sendBtn = document.getElementById('send-btn');
    const chatMessages = document.getElementById('chat-messages');
    
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const uploadStatus = document.getElementById('upload-status');
    
    const healthIndicator = document.getElementById('health-indicator');
    const healthDot = healthIndicator.querySelector('.dot');
    const healthText = healthIndicator.querySelector('.text');
    const chunkCount = document.getElementById('chunk-count');

    // Base API URL (since UI is served from the same server, we can use relative paths)
    const API_BASE = window.location.origin;

    // --- Health Check ---
    async function checkHealth() {
        try {
            const res = await fetch(`${API_BASE}/health`);
            if (res.ok) {
                const data = await res.json();
                healthDot.classList.add('online');
                healthText.textContent = `Online (Gemini: ${data.gemini})`;
                chunkCount.textContent = data.bm25_chunks;
            } else {
                throw new Error("Server returned error");
            }
        } catch (e) {
            healthDot.classList.remove('online');
            healthText.textContent = "Offline / Error";
        }
    }
    
    // Initial health check and then every 30s
    checkHealth();
    setInterval(checkHealth, 30000);

    // --- File Upload ---
    uploadArea.addEventListener('click', () => fileInput.click());
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFileUpload(e.target.files[0]);
        }
    });

    async function handleFileUpload(file) {
        uploadStatus.className = 'status-msg';
        uploadStatus.textContent = `Uploading ${file.name}...`;
        
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const res = await fetch(`${API_BASE}/ingest`, {
                method: 'POST',
                body: formData
            });
            
            const data = await res.json();
            if (res.ok) {
                uploadStatus.classList.add('status-success');
                uploadStatus.textContent = `Success! Indexed ${data.chunks_indexed} chunks.`;
                checkHealth(); // Update chunk count
            } else {
                uploadStatus.classList.add('status-error');
                uploadStatus.textContent = data.detail || 'Upload failed.';
            }
        } catch (err) {
            uploadStatus.classList.add('status-error');
            uploadStatus.textContent = 'Network error during upload.';
        }
        
        // Hide status after 5s
        setTimeout(() => {
            uploadStatus.className = 'status-msg hidden';
        }, 5000);
    }

    // --- Chat Interface ---
    function appendMessage(role, text, sources = [], chunks = [], latency = null) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}`;
        
        const icon = role === 'user' ? 'fa-user' : 'fa-robot';
        
        let contentHtml = `<p>${text.replace(/\n/g, '<br>')}</p>`;
        
        if (latency) {
            contentHtml += `<small style="color:var(--text-muted); font-size:0.7rem; margin-top:10px; display:block;">Latency: ${(latency/1000).toFixed(2)}s</small>`;
        }
        
        if (chunks.length > 0) {
            let chunksHtml = '<div class="sources-container"><div class="sources-header"><i class="fa-solid fa-book-open"></i> Retrieved Context</div>';
            chunks.forEach(chunk => {
                chunksHtml += `
                <div class="chunk">
                    <div class="chunk-meta">
                        <span>Source: ${chunk.metadata.source}</span>
                        <span>Score: ${chunk.rerank_score.toFixed(2)}</span>
                    </div>
                    <div class="chunk-text">${chunk.text}</div>
                </div>`;
            });
            chunksHtml += '</div>';
            contentHtml += chunksHtml;
        }

        msgDiv.innerHTML = `
            <div class="avatar"><i class="fa-solid ${icon}"></i></div>
            <div class="message-content">${contentHtml}</div>
        `;
        
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return msgDiv;
    }

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const query = queryInput.value.trim();
        if (!query) return;
        
        // Append User Query
        appendMessage('user', query);
        queryInput.value = '';
        
        // Loading state
        sendBtn.disabled = true;
        const loadingMsg = appendMessage('assistant', '<span class="loading-dots">Searching knowledge base</span>');
        
        try {
            const res = await fetch(`${API_BASE}/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query, top_k: 5 })
            });
            
            // Remove loading message
            loadingMsg.remove();
            
            const data = await res.json();
            if (res.ok) {
                appendMessage('assistant', data.answer, data.sources, data.chunks, data.latency_ms);
            } else {
                appendMessage('assistant', `Error: ${data.detail || 'Internal Server Error'}`);
            }
        } catch (err) {
            loadingMsg.remove();
            appendMessage('assistant', 'Network error while reaching the server.');
        } finally {
            sendBtn.disabled = false;
            queryInput.focus();
        }
    });
});

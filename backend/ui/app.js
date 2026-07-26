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

    // --- Health Check & Documents ---
    const documentList = document.createElement('div');
    documentList.className = 'document-list';
    document.querySelector('.upload-section').appendChild(documentList);

    async function fetchDocuments() {
        try {
            const res = await fetch(`${API_BASE}/documents`);
            if (res.ok) {
                const data = await res.json();
                documentList.innerHTML = '';
                data.documents.forEach(doc => {
                    const docEl = document.createElement('div');
                    docEl.className = 'doc-item';
                    docEl.innerHTML = `
                        <span><i class="fa-solid fa-file"></i> ${doc}</span>
                        <i class="fa-solid fa-trash delete-doc" data-doc="${doc}"></i>
                    `;
                    documentList.appendChild(docEl);
                });

                document.querySelectorAll('.delete-doc').forEach(btn => {
                    btn.addEventListener('click', async (e) => {
                        const docName = e.target.getAttribute('data-doc');
                        await fetch(`${API_BASE}/documents/${docName}`, { method: 'DELETE' });
                        fetchDocuments();
                        checkHealth();
                    });
                });
            }
        } catch (e) { console.error("Failed to fetch documents"); }
    }

    async function checkHealth() {
        try {
            const res = await fetch(`${API_BASE}/health`);
            if (res.ok) {
                const data = await res.json();
                healthDot.classList.add('online');
                healthText.textContent = `Online (Gemini: ${data.gemini})`;
                chunkCount.textContent = data.bm25_chunks;
                fetchDocuments();
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
    function appendMessage(role, text, sources = [], chunks = [], latency = null, attempts = null) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}`;
        
        const icon = role === 'user' ? 'fa-user' : 'fa-robot';
        
        let contentHtml = `<div class="msg-text">${marked.parse(text)}</div>`;
        
        if (attempts && attempts > 1) {
            contentHtml += `<small style="color:var(--highlight); font-size:0.7rem; margin-top:5px; display:block;">Agentic Retry: Refined search (${attempts} attempts)</small>`;
        }

        if (latency) {
            contentHtml += `<small style="color:var(--text-muted); font-size:0.7rem; margin-top:10px; display:block;">Latency: ${(latency/1000).toFixed(2)}s</small>`;
        }
        
        if (chunks.length > 0) {
            let chunksHtml = '<div class="sources-container"><div class="sources-header"><i class="fa-solid fa-book-open"></i> Retrieved Context</div>';
            chunks.forEach(chunk => {
                let badge = "[TEXT]";
                if (chunk.metadata.chunk_type === 'visual') badge = "[FIGURE]";
                if (chunk.metadata.chunk_type === 'table') badge = "[TABLE]";
                if (chunk.metadata.chunk_type === 'graph') badge = "[GRAPH]";

                chunksHtml += `
                <div class="chunk">
                    <div class="chunk-meta">
                        <span>Source: ${chunk.metadata.source} <b>${badge}</b></span>
                        <span>Score: ${chunk.rerank_score ? chunk.rerank_score.toFixed(2) : 'N/A'}</span>
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

let sessionId = null;

    // Create session on page load
    async function createSession() {
        try {
            const res = await fetch(`${API_BASE}/session`, { method: 'POST' });
            const data = await res.json();
            sessionId = data.session_id;
        } catch(e) { console.error("Failed to create session"); }
    }
    createSession();

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
            // Using fetch with StreamingResponse from FastAPI
            const res = await fetch(`${API_BASE}/query/stream`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query, top_k: 5, session_id: sessionId })
            });

            loadingMsg.remove();
            
            if (!res.ok) {
                appendMessage('assistant', `Error: ${res.statusText}`);
                return;
            }

            const reader = res.body.getReader();
            const decoder = new TextDecoder('utf-8');
            
            const msgDiv = appendMessage('assistant', '');
            const textEl = msgDiv.querySelector('.msg-text');
            let fullAnswer = "";
            let chunksData = [];

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const chunkStr = decoder.decode(value, { stream: true });
                const lines = chunkStr.split('\n');
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.substring(6));
                            if (data.type === 'chunks') {
                                chunksData = data.data;
                            } else if (data.type === 'token') {
                                fullAnswer += data.data;
                                textEl.innerHTML = marked.parse(fullAnswer);
                                chatMessages.scrollTop = chatMessages.scrollHeight;
                            } else if (data.type === 'done') {
                                // Add context after done
                                if (chunksData.length > 0) {
                                    let chunksHtml = '<div class="sources-container"><div class="sources-header"><i class="fa-solid fa-book-open"></i> Retrieved Context</div>';
                                    chunksData.forEach(chunk => {
                                        let badge = "[TEXT]";
                                        if (chunk.metadata.chunk_type === 'visual') badge = "[FIGURE]";
                                        if (chunk.metadata.chunk_type === 'table') badge = "[TABLE]";
                                        if (chunk.metadata.chunk_type === 'graph') badge = "[GRAPH]";

                                        chunksHtml += `
                                        <div class="chunk">
                                            <div class="chunk-meta">
                                                <span>Source: ${chunk.metadata.source} <b>${badge}</b></span>
                                            </div>
                                            <div class="chunk-text">${chunk.text}</div>
                                        </div>`;
                                    });
                                    chunksHtml += '</div>';
                                    const contentEl = msgDiv.querySelector('.message-content');
                                    contentEl.insertAdjacentHTML('beforeend', chunksHtml);
                                }
                            }
                        } catch(e) {}
                    }
                }
            }
        } catch (err) {
            if (loadingMsg.parentNode) loadingMsg.remove();
            appendMessage('assistant', 'Network error while reaching the server.');
        } finally {
            sendBtn.disabled = false;
            queryInput.focus();
        }
    });
});

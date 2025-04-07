document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const chatDisplay = document.getElementById('chat-display');
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');
    const fileUploadButton = document.getElementById('file-upload-button');
    const fileInput = document.getElementById('file-input');
    const attachedFilesContainer = document.getElementById('attached-files-container');
    const attachedFilesList = document.getElementById('attached-files-list');
    const clearFilesButton = document.getElementById('clear-files-button');
    const modelSelect = document.getElementById('model-select');
    const temperatureSlider = document.getElementById('temperature-slider');
    const temperatureValue = document.getElementById('temperature-value');
    const maxTokensInput = document.getElementById('max-tokens-input');
    const reasoningSelect = document.getElementById('reasoning-select');
    const codLimitGroup = document.getElementById('cod-limit-group');
    const codLimitInput = document.getElementById('cod-limit-input');
    const mcpStatusDisplay = document.getElementById('mcp-active-tools');
    const mcpToolButtons = document.querySelectorAll('.mcp-tool-btn');

    // --- State Variables ---
    let currentThreadId = 'default-thread'; // Basic thread management
    let threads = { // Simple in-memory store
        'default-thread': { messages: [] }
    };
    let attachedFiles = []; // { name: string, content: string }[]
    let activatedTools = new Set(['web_search', 'code_execution', 'image_generation']); // Default enabled tools
    let isSending = false; // Prevent multiple sends

    // --- Settings ---
    let MODEL_NAME = modelSelect.value;
    let TEMPERATURE = parseFloat(temperatureSlider.value);
    let MAX_TOKENS = parseInt(maxTokensInput.value);
    let REASONING_METHOD = reasoningSelect.value;
    let COD_WORD_LIMIT = parseInt(codLimitInput.value);

    // --- Initial Setup ---
    renderCurrentThreadMessages(); // Render any potentially saved messages (if implemented)
    updateReasoningControls();
    initMCPToolButtons();
    updateInputHeight(); // Adjust textarea height initially

    // --- Event Listeners ---
    sendButton.addEventListener('click', handleSendMessage);
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault(); // Prevent default newline on Enter
            handleSendMessage();
        }
    });
    messageInput.addEventListener('input', updateInputHeight);

    fileUploadButton.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    clearFilesButton.addEventListener('click', clearAttachedFiles);

    modelSelect.addEventListener('change', (e) => { MODEL_NAME = e.target.value; });
    temperatureSlider.addEventListener('input', (e) => {
        TEMPERATURE = parseFloat(e.target.value);
        temperatureValue.textContent = TEMPERATURE.toFixed(1);
    });
    maxTokensInput.addEventListener('change', (e) => { MAX_TOKENS = parseInt(e.target.value) || 2048; });
    reasoningSelect.addEventListener('change', (e) => {
        REASONING_METHOD = e.target.value;
        updateReasoningControls();
    });
    codLimitInput.addEventListener('change', (e) => { COD_WORD_LIMIT = parseInt(e.target.value) || 15; });

    // --- Functions ---

    // Adjust textarea height dynamically
    function updateInputHeight() {
        messageInput.style.height = 'auto'; // Temporarily shrink
        let scrollHeight = messageInput.scrollHeight;
        let maxHeight = parseInt(window.getComputedStyle(messageInput).maxHeight); // Get max-height from CSS
         messageInput.style.height = Math.min(scrollHeight, maxHeight) + 'px';
    }

    function updateReasoningControls() {
        codLimitGroup.style.display = REASONING_METHOD === 'cod' ? 'flex' : 'none';
    }

    function handleSendMessage() {
        const messageText = messageInput.value.trim();
        if (!messageText && attachedFiles.length === 0) return; // Don't send empty messages
        if (isSending) return; // Prevent concurrent requests

        let fullMessage = messageText;
        if (attachedFiles.length > 0) {
            const fileContext = attachedFiles.map(f => `\n\n--- Attached File: ${f.name} ---\n${f.content}`).join('');
            fullMessage += fileContext;
        }

        sendMessage(fullMessage); // Pass combined message
        messageInput.value = ''; // Clear input
        clearAttachedFiles();    // Clear files after sending
        updateInputHeight(); // Reset input height
    }

     function handleFileSelect(event) {
        const files = event.target.files;
        if (!files) return;

        // Limit number of files if desired (e.g., max 3)
        // const maxFiles = 3;
        // if (files.length + attachedFiles.length > maxFiles) {
        //     alert(`You can attach a maximum of ${maxFiles} files.`);
        //     return;
        // }

        Array.from(files).forEach(file => {
            // Optional: Add size limit check
            // if (file.size > 5 * 1024 * 1024) { // 5MB limit
            //     alert(`File ${file.name} is too large (max 5MB).`);
            //     return;
            // }

            const reader = new FileReader();
            reader.onload = (e) => {
                attachedFiles.push({ name: file.name, content: e.target.result });
                updateAttachedFilesUI();
            };
            reader.onerror = (e) => {
                 console.error("Error reading file:", file.name, e);
                 alert(`Error reading file: ${file.name}`);
            }
            // Read text-based files
            reader.readAsText(file);
        });

        // Clear the input value so the same file can be selected again if removed
        fileInput.value = '';
    }

    function updateAttachedFilesUI() {
        if (attachedFiles.length > 0) {
            attachedFilesList.textContent = attachedFiles.map(f => f.name).join(', ');
            attachedFilesContainer.style.display = 'flex';
        } else {
            attachedFilesContainer.style.display = 'none';
            attachedFilesList.textContent = '';
        }
         updateInputHeight(); // Recalculate height potentially affected by attachment bar
    }

    function clearAttachedFiles() {
        attachedFiles = [];
        updateAttachedFilesUI();
    }

    function addMessageToCurrentThread(content, sender, isPlaceholder = false, timestamp = new Date()) {
        const thread = threads[currentThreadId];
        if (!thread) return; // Should not happen in this basic setup

        const message = {
            content,
            sender,
            isPlaceholder,
            timestamp,
            id: Date.now() + Math.random() // Simple unique ID
        };

        if (isPlaceholder) {
            thread.messages.push(message); // Add placeholder
        } else {
            // Replace the last placeholder or add new message
            const lastMessage = thread.messages[thread.messages.length - 1];
            if (lastMessage && lastMessage.isPlaceholder && lastMessage.sender === sender) {
                thread.messages[thread.messages.length - 1] = message; // Replace
            } else {
                thread.messages.push(message); // Add new
            }
        }

        renderCurrentThreadMessages(); // Update UI
        // Auto-scroll to bottom
        chatDisplay.scrollTop = chatDisplay.scrollHeight;
    }

    function renderCurrentThreadMessages() {
        const thread = threads[currentThreadId];
        if (!thread) {
            chatDisplay.innerHTML = '<p class="no-messages">Start chatting!</p>';
            return;
        }

        chatDisplay.innerHTML = ''; // Clear display

        if (thread.messages.length === 0) {
             chatDisplay.innerHTML = '<p class="no-messages">No messages yet. Send one below!</p>';
             return;
        }

        thread.messages.forEach((msg, index) => {
            const messageDiv = document.createElement('div');
            messageDiv.classList.add('message', msg.sender);
            messageDiv.dataset.messageId = msg.id;

            const contentDiv = document.createElement('div');
            contentDiv.classList.add('message-content');

            if (msg.isStreaming || (msg.sender === 'bot' && msg.thinking)) {
                // Use the streaming renderer for ongoing or complex bot messages
                renderStreamingCOD(contentDiv, msg.content || '', msg.isStreaming);
            } else if (msg.sender === 'bot') {
                 // Render completed bot message potentially with Markdown
                 contentDiv.innerHTML = marked.parse(msg.content || '');
            } else {
                // Plain text for user message
                contentDiv.textContent = msg.content || '';
            }

            const timestampSpan = document.createElement('span');
            timestampSpan.classList.add('timestamp');
            timestampSpan.textContent = msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

            messageDiv.appendChild(contentDiv);
            messageDiv.appendChild(timestampSpan);
            chatDisplay.appendChild(messageDiv);
        });

        // Scroll to bottom after rendering
        chatDisplay.scrollTop = chatDisplay.scrollHeight;
    }

    // --- MCP & Tools Implementation ---

    class ModelContextProtocol {
        constructor() {
            this.tools = {};
            this.activeTool = null;
            this.enabledTools = new Set(); // Track which tools are user-enabled
        }

        registerTool(name, toolObject) {
            this.tools[name] = toolObject;
            console.log(`Tool registered: ${name}`);
        }

        setEnabledTools(enabledToolNames) {
             this.enabledTools = new Set(enabledToolNames);
             console.log("Enabled tools set to:", Array.from(this.enabledTools));
        }

        async invokeTool(toolName, parameters) {
            if (!this.tools[toolName]) {
                throw new Error(`Tool not found: ${toolName}`);
            }
            if (!this.enabledTools.has(toolName)) {
                 throw new Error(`Tool not enabled by user: ${toolName}`);
            }

            this.activeTool = toolName;
            console.log(`Invoking tool: ${toolName} with parameters:`, parameters);

            try {
                const result = await this.tools[toolName].execute(parameters);
                // Include potential logs from code execution
                const responseData = { result: result.output !== undefined ? result.output : result, logs: result.logs };
                console.log(`Tool ${toolName} result:`, responseData);
                return {
                    tool: toolName,
                    result: responseData
                };
            } catch (error) {
                console.error(`Error executing tool ${toolName}:`, error);
                return {
                    tool: toolName,
                    error: error.message
                };
            } finally {
                this.activeTool = null;
            }
        }

        getAvailableTools() {
             // Return only tools that are currently enabled by the user
            return Object.keys(this.tools)
                .filter(name => this.enabledTools.has(name))
                .map(name => ({
                    name,
                    description: this.tools[name].description
                }));
        }
    }

    class WebSearchTool {
        constructor() {
            this.description = "Search the web for current information. Use it for recent events, facts, or topics needing up-to-date data.";
        }

        async execute({ query, numResults = 5 }) {
            if (!query) throw new Error("WebSearchTool requires a 'query' parameter.");
            try {
                const response = await fetch('/api/web-search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query, numResults })
                });
                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({ error: `Search failed with status: ${response.status}` }));
                    throw new Error(errorData.error || `Search failed with status: ${response.status}`);
                }
                return await response.json();
            } catch (error) {
                console.error("Web search tool error:", error);
                throw error; // Re-throw to be caught by invokeTool
            }
        }
    }

    class CodeExecutionTool {
        constructor() {
            this.description = "Execute JavaScript code snippets. Useful for calculations, data manipulation, or testing small algorithms. Only accepts JavaScript.";
            this.supportedLanguages = ['javascript'];
        }

        async execute({ code, language = 'javascript' }) {
             if (!code) throw new Error("CodeExecutionTool requires a 'code' parameter.");
             if (language.toLowerCase() !== 'javascript') {
                return { output: null, logs: [], error: "Only JavaScript execution is currently supported." };
            }

            console.log("Executing code:\n", code);
            try {
                // Use the sandboxed execution
                const result = await this.executeInSandbox(code);
                 console.log("Sandbox result:", result);
                 // Ensure result structure matches expected { output, logs }
                 return { output: result.result, logs: result.logs || [], error: result.error };
            } catch (error) {
                 console.error("Sandbox execution error:", error);
                return { output: null, logs: [], error: error.message, errorType: error.name };
            }
        }

         // Sandboxed code execution using an iframe
         async executeInSandbox(code) {
            return new Promise((resolve, reject) => {
                const iframe = document.createElement('iframe');
                iframe.style.display = 'none';
                iframe.sandbox = 'allow-scripts'; // Basic sandboxing
                document.body.appendChild(iframe);

                const timeoutDuration = 5000; // 5 second timeout
                let executionTimeout = setTimeout(() => {
                     window.removeEventListener('message', messageHandler);
                     document.body.removeChild(iframe);
                     reject(new Error(`Code execution timed out after ${timeoutDuration / 1000} seconds.`));
                }, timeoutDuration);


                const messageHandler = (event) => {
                    // Important: Check origin for security if this were cross-domain
                    // if (event.origin !== window.location.origin) return;

                    if (event.source === iframe.contentWindow) {
                        clearTimeout(executionTimeout); // Clear timeout on message received
                        window.removeEventListener('message', messageHandler);
                        document.body.removeChild(iframe);

                        if (event.data.error) {
                            // Resolve with error structure, don't reject the promise itself here
                            resolve({ error: event.data.error, errorType: event.data.errorType, logs: event.data.logs });
                        } else {
                             resolve({ result: event.data.result, logs: event.data.logs });
                        }
                    }
                };

                window.addEventListener('message', messageHandler);

                // Inject code into iframe with console capture and message posting
                const sandboxContent = `
                    <!DOCTYPE html>
                    <html>
                    <head><title>Sandbox</title></head>
                    <body>
                    <script>
                        const logs = [];
                        const originalConsoleLog = console.log;
                        const originalConsoleError = console.error;
                        const originalConsoleWarn = console.warn;
                        const originalConsoleInfo = console.info;

                        console.log = function() { logs.push({level:'log', args: Array.from(arguments)}); originalConsoleLog.apply(console, arguments); };
                        console.error = function() { logs.push({level:'error', args: Array.from(arguments)}); originalConsoleError.apply(console, arguments); };
                        console.warn = function() { logs.push({level:'warn', args: Array.from(arguments)}); originalConsoleWarn.apply(console, arguments); };
                        console.info = function() { logs.push({level:'info', args: Array.from(arguments)}); originalConsoleInfo.apply(console, arguments); };

                        window.onerror = function(message, source, lineno, colno, error) {
                            logs.push({ level: 'error', args: ['Uncaught Exception:', message, error ? error.stack : ''] });
                            window.parent.postMessage({ error: message, errorType: error ? error.name : 'Error', logs: logs }, '*');
                            return true; // Prevents default browser error handling
                        };

                        try {
                             // Use an IIFE to capture the return value of the last statement
                            const result = (function() {
                                'use strict'; // Enforce stricter parsing and error handling
                                ${code}
                            })();
                            window.parent.postMessage({ result: result, logs: logs }, '*');
                        } catch(e) {
                            logs.push({ level: 'error', args: ['Execution Error:', e.message, e.stack] });
                            window.parent.postMessage({ error: e.message, errorType: e.name, logs: logs }, '*');
                        }
                    <\/script>
                    </body>
                    </html>
                `;

                // Use srcdoc for better security and handling than write/close
                iframe.srcdoc = sandboxContent;

            });
        }
    }


    class ImageGenerationTool {
        constructor() {
            this.description = "Generate images from text descriptions (prompts). Use for visualization or creating image assets.";
        }

        async execute({ prompt, style = 'realistic', size = '1024x1024' }) { // Default to higher res
            if (!prompt) throw new Error("ImageGenerationTool requires a 'prompt' parameter.");
            try {
                const response = await fetch('/api/generate-image', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt, style, size })
                });
                if (!response.ok) {
                     const errorData = await response.json().catch(() => ({ error: `Image generation failed: ${response.status}` }));
                     throw new Error(errorData.error || `Image generation failed: ${response.status}`);
                }
                const data = await response.json();
                // Return the image URL and the prompt used
                 return { imageUrl: data.imageUrl, generatedPrompt: prompt };
            } catch (error) {
                console.error("Image generation tool error:", error);
                throw error;
            }
        }
    }

    // --- Streaming & LLM Call Logic ---

    class StreamController {
        constructor() {
            this.onData = null;
            this.onComplete = null;
            this.onError = null;
            this.accumulatedResponse = "";
            this.currentMessageId = null;
        }

        setHandlers({ onData, onComplete, onError }) {
            this.onData = onData;
            this.onComplete = onComplete;
            this.onError = onError;
        }

        setMessageId(id) {
            this.currentMessageId = id;
        }

        processChunk(chunk) {
            if (!chunk || !this.onData || this.currentMessageId === null) return;
            this.accumulatedResponse += chunk;
            this.onData(chunk, this.accumulatedResponse, this.currentMessageId);
        }

        complete() {
             if (this.onComplete && this.currentMessageId !== null) {
                this.onComplete(this.accumulatedResponse, this.currentMessageId);
            }
             this.currentMessageId = null; // Reset for next message
             this.accumulatedResponse = "";
        }

        error(err) {
             if (this.onError && this.currentMessageId !== null) {
                this.onError(err, this.currentMessageId);
            }
             this.currentMessageId = null; // Reset for next message
             this.accumulatedResponse = "";
        }
    }

    const mcp = new ModelContextProtocol(); // Create a single instance

    // Register client-side tool handlers
    mcp.registerTool('web_search', new WebSearchTool());
    mcp.registerTool('code_execution', new CodeExecutionTool());
    mcp.registerTool('image_generation', new ImageGenerationTool());

    // Main function to call LLM with streaming and MCP tool handling
    async function callLLMWithMCP(userPrompt, options = {}) {
        isSending = true;
        sendButton.disabled = true; // Disable send button while processing

        // Set enabled tools based on current UI selection
        const currentEnabledTools = Array.from(document.querySelectorAll('.mcp-tool-btn.active')).map(btn => btn.getAttribute('data-tool'));
        mcp.setEnabledTools(currentEnabledTools);

        const availableTools = mcp.getAvailableTools();

        // --- Construct System Prompt ---
        let systemPrompt = `You are an advanced AI assistant specialized in research. Think step-by-step.`;

        if (REASONING_METHOD === 'cod') {
            systemPrompt += `\nUse Chain of Draft (COD) reasoning: Produce minimal concise notes (${COD_WORD_LIMIT} words max) for each thinking step. Use mathematical notation if applicable. Separate steps with periods (.). Write your final answer after '####'.`;
        } else {
            systemPrompt += `\nPlease provide a clear and comprehensive answer.`;
        }

        if (availableTools.length > 0) {
            systemPrompt += `\n\nYou have access to the following tools:\n`;
            systemPrompt += availableTools.map(tool => `- ${tool.name}: ${tool.description}`).join('\n');
            systemPrompt += `\n\nTo use a tool, output a **JSON block** like this within your thought process (before the '####'):\n\`\`\`json\n{\n  "tool_name": "${availableTools[0].name}",\n  "parameters": {\n    "param1": "value1",\n    "param2": "value2"\n  }\n}\n\`\`\`\nI will execute the tool and provide the result back to you in a block like <tool_response name="tool_name">...</tool_response>. Wait for the tool response before continuing your thought process or providing the final answer. Only use tools when necessary.`;
        } else {
            systemPrompt += `\n\nNo tools are currently enabled.`
        }

        // --- Prepare message history ---
        const thread = threads[currentThreadId];
        const history = thread.messages
            .filter(msg => !msg.isPlaceholder) // Exclude placeholders
            .slice(-6) // Limit history context (adjust as needed)
            .map(msg => ({
                role: msg.sender === 'bot' ? 'assistant' : 'user',
                content: msg.content // Use the full content including previous COD/tool steps if any
            }));

        // --- Initial Request ---
        const requestParams = {
            model: MODEL_NAME,
            messages: [
                { role: "system", content: systemPrompt },
                ...history, // Add recent history
                { role: "user", content: userPrompt }
            ],
            temperature: TEMPERATURE,
            max_tokens: MAX_TOKENS,
            stream: true // Always stream
        };

        try {
            // Add placeholder message and get its ID
             addMessageToCurrentThread("Thinking...", "bot", true);
             const placeholderMsg = thread.messages[thread.messages.length - 1];
             const messageId = placeholderMsg.id;

             const streamController = new StreamController();
             streamController.setMessageId(messageId);

             // Set up streaming handlers
             streamController.setHandlers({
                onData: (chunk, accumulated, msgId) => {
                    // Update the placeholder message content in real-time
                    const targetMsg = thread.messages.find(m => m.id === msgId);
                    if (targetMsg) {
                        targetMsg.content = accumulated;
                        targetMsg.isStreaming = true;
                        // Re-render only the specific message being updated for efficiency
                        const messageDiv = chatDisplay.querySelector(`.message[data-message-id="${msgId}"] .message-content`);
                        if(messageDiv) {
                             renderStreamingCOD(messageDiv, accumulated, true);
                             chatDisplay.scrollTop = chatDisplay.scrollHeight; // Keep scrolled down
                        }
                    }
                },
                onComplete: (finalResponse, msgId) => {
                    console.log("Stream complete. Final response:", finalResponse);
                    const targetMsgIndex = thread.messages.findIndex(m => m.id === msgId);
                    if (targetMsgIndex !== -1) {
                        const processed = processBotMessage(finalResponse);
                        thread.messages[targetMsgIndex] = {
                            ...thread.messages[targetMsgIndex], // Keep id, sender, timestamp
                            content: finalResponse,
                            isPlaceholder: false,
                            isStreaming: false,
                            reasoningMethod: getMCPReasoningMethod(),
                            thinking: processed.thinking,
                            answer: processed.answer,
                            thinkingWordCount: processed.thinkingWordCount,
                            answerWordCount: processed.answerWordCount,
                            usedTools: processed.usedTools // Track tools used in this response
                        };
                         renderCurrentThreadMessages(); // Full re-render to finalize
                    }
                    isSending = false; // Re-enable sending
                    sendButton.disabled = false;
                },
                onError: (error, msgId) => {
                    console.error("Streaming error:", error);
                     const targetMsgIndex = thread.messages.findIndex(m => m.id === msgId);
                     if (targetMsgIndex !== -1) {
                        thread.messages[targetMsgIndex] = {
                            ...thread.messages[targetMsgIndex],
                            content: `Error: ${error.message}`,
                            isPlaceholder: false,
                            isStreaming: false,
                        };
                         renderCurrentThreadMessages(); // Show error
                    }
                    isSending = false; // Re-enable sending
                    sendButton.disabled = false;
                }
            });

            // Start the streaming process, potentially handling tools
            await streamLLMAndHandleTools(requestParams, mcp, streamController);

        } catch (error) {
            console.error("LLM Call initiation error:", error);
             addMessageToCurrentThread(`Error: ${error.message}`, "bot", false); // Show error directly
            isSending = false; // Re-enable sending
            sendButton.disabled = false;
        }
    }


    // Handles the stream and potential tool calls within it
    async function streamLLMAndHandleTools(initialRequestParams, mcpInstance, streamController) {
        let currentMessages = [...initialRequestParams.messages];
        let accumulatedContentForTurn = "";
        let usedToolsInTurn = []; // Track tools used in this specific turn

        try {
            const response = await fetch('/api/llm-stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ...initialRequestParams, messages: currentMessages }) // Send current message history
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: `API request failed: ${response.status}` }));
                throw new Error(errorData.error || `API request failed: ${response.status}`);
            }
            if (!response.body) {
                 throw new Error("Response body is null");
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let toolCallDetected = false;
            let toolCallJson = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                // Process line by line for SSE
                const lines = buffer.split('\n');
                buffer = lines.pop() || ''; // Keep incomplete line in buffer

                for (const line of lines) {
                     if (line.startsWith('data: ')) {
                        const data = line.slice(6);
                        if (data === '[DONE]') {
                            // If DONE is received but we were expecting a tool call, it's an error or hallucination
                             if (toolCallDetected) {
                                 console.warn("Stream ended with [DONE] while parsing a tool call JSON.");
                                 // Append the partial JSON as plain text? Or error?
                                 streamController.processChunk(toolCallJson);
                                 toolCallDetected = false; // Reset
                            }
                            continue; // Move to next line or break loop if buffer empty
                        }

                         try {
                            const parsedData = JSON.parse(data);
                            if (parsedData.error) {
                                throw new Error(`LLM API Error: ${parsedData.error.message || JSON.stringify(parsedData.error)}`);
                            }

                             if (parsedData.choices && parsedData.choices[0].delta?.content) {
                                const chunk = parsedData.choices[0].delta.content;
                                accumulatedContentForTurn += chunk; // Accumulate all content for this turn

                                // --- Tool Call Detection (Improved) ---
                                // Look for the start of a JSON block ```json
                                if (!toolCallDetected && accumulatedContentForTurn.includes('```json')) {
                                     // Check if it looks like our tool call format immediately after
                                     const jsonStartIndex = accumulatedContentForTurn.lastIndexOf('```json');
                                     const potentialJson = accumulatedContentForTurn.substring(jsonStartIndex + 7).trimStart();
                                     if (potentialJson.startsWith('{') && potentialJson.includes('"tool_name"')) {
                                         console.log("Potential tool call JSON detected...");
                                         toolCallDetected = true;
                                         // Don't stream the ```json marker itself yet
                                         streamController.processChunk(accumulatedContentForTurn.substring(0, jsonStartIndex)); // Stream content before JSON
                                         toolCallJson = potentialJson; // Start accumulating JSON content
                                         accumulatedContentForTurn = ""; // Reset accumulator for content *after* JSON start
                                         continue; // Skip normal chunk processing for this specific chunk
                                     }
                                }

                                if (toolCallDetected) {
                                     toolCallJson += chunk; // Accumulate the JSON part
                                     // Check if the JSON block is complete
                                     if (toolCallJson.includes('```')) {
                                         console.log("Tool call JSON end detected.");
                                         const jsonEndIndex = toolCallJson.lastIndexOf('```');
                                         const finalJson = toolCallJson.substring(0, jsonEndIndex).trim();
                                         const contentAfterJson = toolCallJson.substring(jsonEndIndex + 3); // Content after ```

                                          toolCallDetected = false; // Reset detection

                                         try {
                                             const toolInvocation = JSON.parse(finalJson);
                                             if (toolInvocation.tool_name && toolInvocation.parameters) {
                                                 // Valid tool call structure
                                                 console.log("Executing Tool:", toolInvocation.tool_name);

                                                 // Render the call in the UI *before* execution
                                                 const toolCallMarker = `\n\`\`\`json\n${JSON.stringify(toolInvocation, null, 2)}\n\`\`\`\n`;
                                                 streamController.processChunk(toolCallMarker); // Show the formatted call

                                                 // --- Execute the tool ---
                                                 const toolResult = await mcpInstance.invokeTool(toolInvocation.tool_name, toolInvocation.parameters);
                                                 usedToolsInTurn.push(toolInvocation.tool_name); // Track usage

                                                 // --- Format tool response ---
                                                 let toolResponseContent;
                                                 if (toolResult.error) {
                                                      toolResponseContent = JSON.stringify({ error: toolResult.error }, null, 2);
                                                 } else {
                                                      toolResponseContent = JSON.stringify(toolResult.result, null, 2);
                                                 }
                                                 const toolResponseMarker = `<tool_response name="${toolInvocation.tool_name}">\n${toolResponseContent}\n</tool_response>\n`;

                                                 // --- Continue the LLM call with the tool response ---
                                                 // Add the assistant's partial thought + tool call, and the tool response to the history
                                                 currentMessages.push({ role: 'assistant', content: streamController.accumulatedResponse + toolCallMarker }); // Add LLM's output up to tool call
                                                 currentMessages.push({ role: 'tool', tool_call_id: "N/A", // DeepSeek might not use IDs like OpenAI
                                                                          name: toolInvocation.tool_name,
                                                                          content: toolResponseContent }); // Add actual tool result

                                                 // Display the response marker in the UI immediately
                                                 streamController.processChunk(toolResponseMarker);

                                                 console.log("Continuing LLM call with tool response...");
                                                 // Recurse or loop: make a new stream request with updated history
                                                 // For simplicity here, we restart the stream with updated context.
                                                 // A more complex implementation might try to resume the *same* stream if the API supports it.
                                                 await streamLLMAndHandleTools({ ...initialRequestParams, messages: currentMessages }, mcpInstance, streamController);
                                                 return; // Exit this loop level as the recursive call handles the rest

                                             } else {
                                                 console.warn("Parsed JSON, but not a valid tool call structure:", finalJson);
                                                  streamController.processChunk(chunk); // Process as normal text
                                             }
                                         } catch (e) {
                                             console.error("Failed to parse JSON or execute tool:", e);
                                             // Append the raw block as text if parsing failed
                                             streamController.processChunk('\n```json\n' + finalJson + '\n```\n');
                                              streamController.processChunk(contentAfterJson); // Stream any content after ```
                                         }
                                         toolCallJson = ""; // Reset json accumulator
                                     }
                                     // Don't process this chunk normally if we are accumulating JSON
                                     continue;
                                }

                                // If not detecting/parsing a tool call, process the chunk normally
                                streamController.processChunk(chunk);
                             }
                         } catch (e) {
                            console.error("Error processing stream data chunk:", e, "Data:", data);
                            // Maybe signal error to UI? For now, just log.
                         }
                    } // End startsWith('data: ')
                } // End line processing loop
            } // End while(true) loop

            // If loop finishes without tool execution interrupting, complete the stream
            console.log("Stream finished naturally.");
            streamController.complete(); // Signal completion with the final accumulated content

        } catch (error) {
            console.error('Error during streaming or tool handling:', error);
            streamController.error(error); // Signal error
        } finally {
             // Ensure sending state is reset even if errors occur during the stream
             if (!toolCallDetected) { // Only reset if not in the middle of a tool call continuation
                 isSending = false;
                 sendButton.disabled = false;
             }
        }
    }


    // --- UI Rendering for Streaming & COD ---

    function renderStreamingCOD(messageContentDiv, content, isStreaming) {
        // Clear previous content inside the message div
        messageContentDiv.innerHTML = '';

        // Helper function to safely format JSON
        function formatJsonSafely(jsonString) {
             try {
                return JSON.stringify(JSON.parse(jsonString), null, 2);
             } catch {
                return jsonString; // Return original if not valid JSON
             }
        }

        // 1. Split content into thought/tool parts and final answer part
        let thinkingPart = content;
        let answerPart = '';
        const separator = '####';
        const separatorIndex = content.lastIndexOf(separator); // Use lastIndexOf in case '####' appears in thoughts

        if (separatorIndex !== -1) {
            thinkingPart = content.substring(0, separatorIndex).trim();
            answerPart = content.substring(separatorIndex + separator.length).trim();
        }

        // 2. Render Thinking/Tool Part
        if (thinkingPart) {
            const thinkingContainer = document.createElement('div');
            thinkingContainer.className = 'cod-thinking';

            const thinkingLabel = document.createElement('div');
            thinkingLabel.className = 'thinking-label';
            thinkingLabel.textContent = 'Assistant Reasoning'; // More general label
            thinkingContainer.appendChild(thinkingLabel);

            const stepsContainer = document.createElement('div');
            stepsContainer.className = 'steps-container';
            thinkingContainer.appendChild(stepsContainer);

            // Regex to split by steps (.), tool calls (```json...```), and tool responses (<tool_response>...</tool_response>)
            // This regex tries to capture these blocks or simple text separated by periods.
             const thinkingRegex = /(```json[\s\S]*?```|<tool_response name="[^"]+">[\s\S]*?<\/tool_response>|[^.]+?\.)/gs;
             let thinkingMatch;
             let stepCounter = 0;
             let lastIndex = 0;

             while ((thinkingMatch = thinkingRegex.exec(thinkingPart)) !== null) {
                stepCounter++;
                const stepContent = thinkingMatch[0].trim();
                lastIndex = thinkingRegex.lastIndex;

                if (stepContent.startsWith('```json')) {
                    addToolCallUI(stepsContainer, stepContent, stepCounter);
                } else if (stepContent.startsWith('<tool_response')) {
                    addToolResponseUI(stepsContainer, stepContent, stepCounter);
                } else if (stepContent) { // Regular COD step
                    addCodStepUI(stepsContainer, stepContent, stepCounter);
                }
             }

             // Add any remaining text after the last matched delimiter as a final step
             const remainingThinkingText = thinkingPart.substring(lastIndex).trim();
             if (remainingThinkingText) {
                 stepCounter++;
                 addCodStepUI(stepsContainer, remainingThinkingText, stepCounter);
             }


            messageContentDiv.appendChild(thinkingContainer);
        }

        // 3. Render Final Answer Part (if available)
        if (answerPart) {
            const answerDiv = document.createElement('div');
            answerDiv.className = 'final-answer';

            const answerLabel = document.createElement('div');
            answerLabel.className = 'final-answer-label';
            answerLabel.textContent = 'Final Answer';
            answerDiv.appendChild(answerLabel);

            const answerContent = document.createElement('div');
            // Use marked to render Markdown in the final answer
            answerContent.innerHTML = marked.parse(answerPart);
            // Sanitize HTML if needed using a library like DOMPurify
            answerDiv.appendChild(answerContent);

            messageContentDiv.appendChild(answerDiv);
        }

        // 4. Add Thinking Indicator if still streaming and no final answer yet
        if (isStreaming && separatorIndex === -1) {
            let indicatorContainer = messageContentDiv.querySelector('.cod-thinking .steps-container');
            // If no thinking container yet, create a minimal one
            if (!indicatorContainer) {
                 const thinkingContainer = document.createElement('div');
                 thinkingContainer.className = 'cod-thinking';
                 const stepsContainer = document.createElement('div');
                 stepsContainer.className = 'steps-container';
                 thinkingContainer.appendChild(stepsContainer);
                 messageContentDiv.appendChild(thinkingContainer);
                 indicatorContainer = stepsContainer;
            }

            const thinkingIndicator = document.createElement('div');
            thinkingIndicator.className = 'thinking-indicator';
            // thinkingIndicator.textContent = 'Thinking'; // Text is added via ::after pseudo-element
             indicatorContainer.appendChild(thinkingIndicator);
        }

         // Render images found in the final answer markdown
         renderImagesInContent(messageContentDiv);
    }

    function addCodStepUI(container, content, index) {
        const stepEl = document.createElement('div');
        stepEl.className = 'step cod-step';

        const stepNumber = document.createElement('span');
        stepNumber.className = 'step-number';
        stepNumber.textContent = `${index}.`; // Add dot to number

        const stepContentEl = document.createElement('span');
        stepContentEl.className = 'step-content';
        stepContentEl.textContent = content.endsWith('.') ? content.slice(0,-1) : content; // Remove trailing dot if present

        stepEl.appendChild(stepNumber);
        stepEl.appendChild(stepContentEl);
        container.appendChild(stepEl);

        // Animate step appearing
        setTimeout(() => { stepEl.classList.add('visible'); }, 5 * index); // Faster animation
    }


    function addToolCallUI(container, content, index) {
        const toolCallDiv = document.createElement('div');
        toolCallDiv.className = 'tool-call-container cod-step'; // Treat as a step visually

        let toolName = 'unknown tool';
        let formattedParams = content; // Default to raw content

        // Extract tool name and parameters from the JSON block
        const jsonMatch = content.match(/```json([\s\S]*?)```/);
        if (jsonMatch && jsonMatch[1]) {
            try {
                const parsedJson = JSON.parse(jsonMatch[1].trim());
                toolName = parsedJson.tool_name || toolName;
                 // Pretty print the parameters JSON
                 formattedParams = JSON.stringify(parsedJson, null, 2);
            } catch (e) {
                console.warn("Could not parse tool call JSON for UI:", e);
                formattedParams = jsonMatch[1].trim(); // Show raw JSON if parse fails
            }
        }

        // Create header
        const header = document.createElement('div');
        header.className = 'tool-call-header';
        const iconSvg = getToolIcon(toolName);
        header.innerHTML = `${iconSvg} <span>Calling: ${toolName}</span>`;
        toolCallDiv.appendChild(header);

        // Add content (formatted JSON)
        const codeContent = document.createElement('pre');
        codeContent.className = 'tool-call-content';
        codeContent.textContent = formattedParams;
        toolCallDiv.appendChild(codeContent);

        container.appendChild(toolCallDiv);
        setTimeout(() => { toolCallDiv.classList.add('visible'); }, 5 * index);
    }

     function addToolResponseUI(container, content, index) {
        const responseDiv = document.createElement('div');
        responseDiv.className = 'tool-response-container cod-step'; // Treat as a step visually

        let toolName = 'unknown tool';
        let responseData = content; // Default to raw content

        // Extract tool name and content from the response block
         const responseMatch = content.match(/<tool_response name="([^"]+)">([\s\S]*?)<\/tool_response>/);
        if (responseMatch) {
             toolName = responseMatch[1];
             responseData = responseMatch[2].trim();
        }

        // Create header
        const header = document.createElement('div');
        header.className = 'tool-call-header';
         // Use a generic response icon or reuse tool icon
         const iconSvg = getToolIcon(toolName); // Or a specific response icon
        header.innerHTML = `${iconSvg} <span>Response from: ${toolName}</span>`;
        responseDiv.appendChild(header);

        // Add content
        const responseContent = document.createElement('div');
        responseContent.className = 'tool-call-content'; // Reuse styling

        try {
            // Try formatting as JSON
            const jsonData = JSON.parse(responseData);
             // Special handling for image generation response
             if (toolName === 'image_generation' && jsonData.result?.imageUrl) {
                 responseContent.innerHTML = `
                     <p>Image generated successfully:</p>
                     <img src="${jsonData.result.imageUrl}" alt="Generated image for prompt: ${jsonData.result.generatedPrompt}" style="max-width: 200px; height: auto; border-radius: 4px; margin-top: 5px;">
                     <p style="font-size: 0.8em; margin-top: 5px;">Prompt: ${jsonData.result.generatedPrompt}</p>
                 `;
             } else if (toolName === 'code_execution') {
                  // Nicer formatting for code execution results
                  let outputHTML = `<p><strong>Output:</strong></p><pre>${JSON.stringify(jsonData.result?.output, null, 2) || 'null'}</pre>`;
                  if (jsonData.result?.logs && jsonData.result.logs.length > 0) {
                      outputHTML += `<p style="margin-top: 10px;"><strong>Logs:</strong></p><pre>${jsonData.result.logs.map(log => `[${log.level}] ${log.args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : arg).join(' ')}`).join('\n')}</pre>`;
                  }
                   if (jsonData.error) { // Show error prominently if present
                       outputHTML += `<p style="margin-top: 10px; color: var(--error);"><strong>Error:</strong> ${jsonData.error}</p>`;
                  }
                   responseContent.innerHTML = outputHTML;

             } else {
                // Default: Pretty print JSON
                responseContent.innerHTML = `<pre>${JSON.stringify(jsonData, null, 2)}</pre>`;
             }
        } catch (e) {
            // If not valid JSON or specific format, just show the text
            responseContent.textContent = responseData;
        }


        responseDiv.appendChild(responseContent);
        container.appendChild(responseDiv);
        setTimeout(() => { responseDiv.classList.add('visible'); }, 5 * index);
    }

     // Helper to get SVG icon based on tool name
     function getToolIcon(toolName) {
         switch (toolName) {
            case 'web_search':
                return '<svg xmlns="[http://www.w3.org/2000/svg](http://www.w3.org/2000/svg)" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>';
            case 'code_execution':
                return '<svg xmlns="[http://www.w3.org/2000/svg](http://www.w3.org/2000/svg)" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>';
            case 'image_generation':
                return '<svg xmlns="[http://www.w3.org/2000/svg](http://www.w3.org/2000/svg)" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>';
            default: // Generic tool icon
                return '<svg xmlns="[http://www.w3.org/2000/svg](http://www.w3.org/2000/svg)" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>';
         }
     }

     // Helper to render images found within markdown content (e.g., ![alt](url))
     function renderImagesInContent(contentDiv) {
         contentDiv.querySelectorAll('img').forEach(img => {
             // Optional: Add error handling or styling
             img.onerror = () => { img.alt = "Failed to load image"; img.style.display = 'none'; };
             img.style.maxWidth = '100%'; // Ensure images are responsive
             img.style.height = 'auto';
             img.style.borderRadius = 'var(--radius-sm)';
             img.style.marginTop = '10px';
         });
     }


    // --- Utility Functions ---

    function processBotMessage(content) {
        // Basic processing to extract parts, assuming COD format when applicable
        let thinking = content;
        let answer = '';
        let thinkingWordCount = 0;
        let answerWordCount = 0;
        const usedTools = []; // Extract tool usage if possible from content analysis

        if (REASONING_METHOD === 'cod') {
            const separatorIndex = content.lastIndexOf('####');
            if (separatorIndex !== -1) {
                thinking = content.substring(0, separatorIndex).trim();
                answer = content.substring(separatorIndex + 4).trim();
            } else {
                // Assume entire message is thinking if separator not found
                thinking = content;
                answer = '';
            }
             thinkingWordCount = thinking.split(/\s+/).filter(Boolean).length;
             answerWordCount = answer.split(/\s+/).filter(Boolean).length;
        } else {
             // Standard method - whole content is the answer
             thinking = '';
             answer = content;
             answerWordCount = answer.split(/\s+/).filter(Boolean).length;
        }

         // Simple check for tool markers in the thinking part
         if (thinking.includes('```json') && thinking.includes('"tool_name"')) usedTools.push('tool_call_detected');
         if (thinking.includes('<tool_response name=')) usedTools.push('tool_response_detected');


        return { thinking, answer, thinkingWordCount, answerWordCount, usedTools };
    }

    function getMCPReasoningMethod() {
        let reasoningInfo = REASONING_METHOD.toUpperCase();
        if (REASONING_METHOD === 'cod') {
            reasoningInfo += `-${COD_WORD_LIMIT}`;
        }
        // Add MCP indicator only if tools are active
        if (activatedTools.size > 0) {
             reasoningInfo += "-MCP";
        }
        return reasoningInfo;
    }


    // Initialize MCP Tool Buttons and Status
    function initMCPToolButtons() {
        // Load saved preferences
        const savedTools = localStorage.getItem('activatedTools');
        if (savedTools) {
            try {
                 activatedTools = new Set(JSON.parse(savedTools));
            } catch (e) {
                 console.error("Failed to parse saved tools, using defaults.", e);
                 activatedTools = new Set(['web_search', 'code_execution', 'image_generation']); // Default if parse fails
                 localStorage.setItem('activatedTools', JSON.stringify(Array.from(activatedTools)));
            }
        } else {
            // If no saved state, save the default
             localStorage.setItem('activatedTools', JSON.stringify(Array.from(activatedTools)));
        }


        mcpToolButtons.forEach(button => {
            const toolName = button.getAttribute('data-tool');
            // Set initial active state from loaded/default set
            if (activatedTools.has(toolName)) {
                button.classList.add('active');
            } else {
                button.classList.remove('active');
            }

            button.addEventListener('click', () => {
                button.classList.toggle('active');
                if (button.classList.contains('active')) {
                    activatedTools.add(toolName);
                } else {
                    activatedTools.delete(toolName);
                }
                updateMCPStatus();
                // Save preferences
                localStorage.setItem('activatedTools', JSON.stringify(Array.from(activatedTools)));
            });
        });

        updateMCPStatus(); // Set initial status text
    }

    // Update MCP status display text and indicator color
    function updateMCPStatus() {
         const numActive = activatedTools.size;
         const totalTools = mcpToolButtons.length;
         const statusElement = mcpStatusDisplay; // The <span> inside .mcp-status

         statusElement.classList.remove('all', 'some', 'none'); // Reset classes

        if (numActive === 0) {
            statusElement.textContent = 'No tools enabled';
             statusElement.classList.add('none');
        } else if (numActive === totalTools) {
            statusElement.textContent = 'All tools enabled';
            statusElement.classList.add('all'); // 'all' implicitly uses --success via ::before default
        } else {
            const activeNames = Array.from(activatedTools).map(tool => {
                const btn = document.querySelector(`.mcp-tool-btn[data-tool="${tool}"] span`);
                return btn ? btn.textContent : tool; // Get readable name from button
            });
            statusElement.textContent = `Enabled: ${activeNames.join(', ')}`;
             statusElement.classList.add('some');
        }
    }

}); // End DOMContentLoaded

document.addEventListener("DOMContentLoaded", () => {
  const chatBox = document.getElementById("chat-box");
  const userInput = document.getElementById("user-input");
  const historyList = document.getElementById("chat-history-list");
  const newChatBtn = document.getElementById("new-chat-btn");
  const logoutBtn = document.getElementById("logout-btn");

  let currentConvId = null; // Store current conversation id
  let firstMessageSent = false;

  async function sendMessage() {
    // Remove welcome message on first send
    if (!firstMessageSent) {
        const welcomeMsg = document.getElementById('welcome-msg');
        if (welcomeMsg) welcomeMsg.remove();
        firstMessageSent = true;
    }

    let message = userInput.value.trim();
    if (!message) return;
    appendMessage("You", message, "user-msg");
    appendHistory("You", message); // optional: can update history preview

    userInput.value = "";
    try {
      let headers = { "Content-Type": "application/json", "Authorization": `Bearer ${localStorage.getItem("jwt")}` };
      const response = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers,
        body: JSON.stringify({ message: message, conversation_id: currentConvId })
      });
      const data = await response.json();
      if (response.ok) {
        appendMessage("Assistant", data.response, "assistant-msg");
        appendHistory("Assistant", data.response);
        // Update current conversation id if new chat
        currentConvId = data.conversation_id;
        fetchConversations();
      } else {
        appendMessage("Error", data.detail, "error-msg");
      }
    } catch (error) {
      appendMessage("Error", error.message, "error-msg");
    }
    chatBox.scrollTop = chatBox.scrollHeight;
  }

  // New Chat: resets the chat window and sets a new conversation id
  function newChat() {
    chatBox.innerHTML = "";
    userInput.value = "";
    currentConvId = null;
    firstMessageSent = false;
    // Show welcome message again
    if (!document.getElementById('welcome-msg')) {
        const welcome = document.createElement('div');
        welcome.id = 'welcome-msg';
        welcome.className = 'text-gray-400 italic p-4 text-lg text-center bg-gray-900 rounded-md my-8 mx-auto max-w-xl shadow-lg';
        welcome.innerHTML = "👋 Hi, I’m your <span class='font-semibold text-blue-300'>Mental Health Buddy</span>.<br>What’s on your mind today?";
        chatBox.appendChild(welcome);
    }
  }

  // Fetch conversation history for the logged in user
  async function fetchConversations() {
    try {
      const res = await fetch("http://127.0.0.1:8000/conversations", {
        headers: { "Authorization": `Bearer ${localStorage.getItem("jwt")}` }
      });
      if (res.ok) {
        const convs = await res.json();
        renderChatHistory(convs);
      }
    } catch (error) {
      console.error("Error fetching conversations:", error);
    }
  }

  // Load full conversation messages for a given conversation id
  async function loadConversation(conv_id) {
    try {
      const res = await fetch(`http://127.0.0.1:8000/conversation/${conv_id}`, {
        headers: { "Authorization": `Bearer ${localStorage.getItem("jwt")}` }
      });
      if (res.ok) {
        const messages = await res.json();
        chatBox.innerHTML = "";
        messages.forEach(msg => {
          appendMessage(msg.sender, msg.message, msg.sender === "User" ? "user-msg" : "assistant-msg");
        });
        currentConvId = conv_id;  // set current conversation
      }
    } catch (error) {
      console.error("Error loading conversation:", error);
    }
  }

  // Function to use Web Speech API for voice input remains unchanged
  function startVoice() {
    if (!("webkitSpeechRecognition" in window)) {
      alert("Your browser does not support the Web Speech API.");
      return;
    }
    let recognition = new webkitSpeechRecognition();
    recognition.lang = "en-US";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onresult = (event) => {
      let transcript = event.results[0][0].transcript;
      userInput.value = transcript;
      sendMessage();
    };

    recognition.onerror = (event) => {
      console.error("Speech recognition error:", event.error);
      appendMessage("Error", event.error, "error-msg");
    };

    recognition.start();
  }

  // Function to append a message in the chat window
  function appendMessage(sender, message, cssClass) {
    // Ensure message is a string
    if (typeof message !== "string") {
      message = JSON.stringify(message);
    }
    const p = document.createElement("p");
    p.className = cssClass;
    p.style.whiteSpace = "pre-wrap";  // preserve whitespaces and line breaks

    // Create the read aloud button
    const readBtn = document.createElement("button");
    readBtn.innerText = "🔊";
    readBtn.title = "Read aloud";
    readBtn.className = "read-aloud-btn";
    readBtn.style.marginLeft = "10px";
    readBtn.onclick = function(e) {
      e.stopPropagation();
      readAloud(message);
    };

    // Message content
    p.innerHTML = `<strong>${sender}:</strong> ${message.replace(/\n/g, '<br>')}`;
    p.appendChild(readBtn);

    chatBox.appendChild(p);
  }

  // Add this function to handle reading aloud
  function readAloud(text) {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel(); // Stop any previous speech
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1; // Normal speed
      window.speechSynthesis.speak(utterance);
    } else {
      alert("Sorry, your browser does not support speech synthesis.");
    }
  }

  // Function to append messages to the chat history sidebar
  function appendHistory(sender, message) {
    const li = document.createElement("li");
    li.innerHTML = `<strong>${sender}:</strong> ${message}`;
    historyList.appendChild(li);
  }

  // Helper to create a chat history item with delete icon
  function createChatHistoryItem(conversation) {
    const li = document.createElement('li');
    li.className = 'border-b border-gray-600 pb-1 text-sm text-white flex justify-between items-center';
    li.dataset.conversationId = conversation.conversation_id;
    // Chat title clickable
    const titleSpan = document.createElement('span');
    titleSpan.textContent = conversation.title;
    titleSpan.className = 'cursor-pointer flex-1';
    // Delete icon
    const deleteBtn = document.createElement('button');
    deleteBtn.innerHTML = '🗑️';
    deleteBtn.className = 'ml-2 text-red-400 hover:text-red-600 delete-chat-btn';
    deleteBtn.title = 'Delete chat';
    deleteBtn.onclick = function(e) {
        e.stopPropagation();
        const convId = li.dataset.conversationId;
        if (confirm('Delete this chat?')) {
            deleteConversation(convId, li);
        }
    };
    li.appendChild(titleSpan);
    li.appendChild(deleteBtn);
    // Optionally, clicking the title loads the chat
    titleSpan.onclick = function() {
        loadConversation(conversation.conversation_id);
    };
    return li;
  }

  // Function to delete conversation via backend
  function deleteConversation(conversationId, liElement) {
    const token = localStorage.getItem('jwt');
    fetch(`http://127.0.0.1:8000/conversation/${conversationId}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${token}` }
    })
    .then(res => {
        if (!res.ok) throw new Error('Failed to delete');
        // Remove from UI
        liElement.remove();
    })
    .catch(() => alert('Failed to delete chat.'));
  }

  // Update chat history rendering to use the new helper
  function renderChatHistory(conversations) {
    const list = document.getElementById('chat-history-list');
    list.innerHTML = '';
    conversations.forEach(conv => {
        list.appendChild(createChatHistoryItem(conv));
    });
  }

  // Expose functions to the global scope for usage from HTML
  window.sendMessage = sendMessage;
  window.startVoice = startVoice;

  // Attach "New Chat" button event listener
  newChatBtn.addEventListener("click", newChat);

  // Logout button event listener
  if (logoutBtn) {
    logoutBtn.addEventListener("click", () => {
      // Clear any stored JWT/authentication data
      localStorage.removeItem("jwt");
      localStorage.removeItem("username");
      // Optionally clear the chat area or update UI as needed
      // Redirect the user to login page
      window.location.href = "login.html";
    });
  }

  // On page load, fetch conversation history if JWT exists
  if (localStorage.getItem("jwt")) {
    fetchConversations();
  }
});
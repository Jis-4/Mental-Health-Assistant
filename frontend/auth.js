document.addEventListener("DOMContentLoaded", () => {
  const API = "http://127.0.0.1:8000"; // Backend runs on port 8000

  const signupForm = document.getElementById("signup-form");
  const loginForm = document.getElementById("login-form");

  if (signupForm) {
    signupForm.onsubmit = async (e) => {
      e.preventDefault();
      const username = document.getElementById("signup-username").value;
      const email = document.getElementById("signup-email").value;
      const password = document.getElementById("signup-password").value;
      const res = await fetch(`${API}/auth/signup`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, email, password })
      });
      const data = await res.json();
      if (res.ok) {
        // Store JWT in localStorage and redirect to main interface
        localStorage.setItem("jwt", data.access_token);
        localStorage.setItem("username", data.username);
        window.location.href = "index.html";
      } else {
        alert(data.detail || "Signup failed");
      }
    };
  }

  if (loginForm) {
    loginForm.onsubmit = async (e) => {
      e.preventDefault();
      const email = document.getElementById("login-email").value;
      const password = document.getElementById("login-password").value;
      const res = await fetch(`${API}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password })
      });
      const data = await res.json();
      if (res.ok) {
        localStorage.setItem("jwt", data.access_token);
        localStorage.setItem("username", data.username);
        window.location.href = "index.html";
      } else {
        alert(data.detail || "Login failed");
      }
    };
  }
});
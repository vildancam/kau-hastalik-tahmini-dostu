const state = {
    muted: localStorage.getItem("muted") === "true",
    audioContext: null,
};

function playTone(type = "click") {
    if (state.muted) return;
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!AudioContext) return;
    state.audioContext = state.audioContext || new AudioContext();
    const ctx = state.audioContext;
    const oscillator = ctx.createOscillator();
    const gain = ctx.createGain();
    const frequencies = {click: 360, success: 620, error: 160};
    oscillator.frequency.value = frequencies[type] || frequencies.click;
    oscillator.type = "sine";
    gain.gain.setValueAtTime(0.045, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.16);
    oscillator.connect(gain).connect(ctx.destination);
    oscillator.start();
    oscillator.stop(ctx.currentTime + 0.16);
}

const themeToggle = document.getElementById("themeToggle");
const savedTheme = localStorage.getItem("theme") || "light";
document.documentElement.setAttribute("data-theme", savedTheme);

themeToggle?.addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme");
    const next = current === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("theme", next);
    playTone("click");
});

const muteToggle = document.getElementById("muteToggle");
function renderMuteIcon() {
    if (!muteToggle) return;
    muteToggle.innerHTML = state.muted
        ? '<i class="bi bi-volume-mute"></i>'
        : '<i class="bi bi-volume-up"></i>';
}
renderMuteIcon();
muteToggle?.addEventListener("click", () => {
    state.muted = !state.muted;
    localStorage.setItem("muted", String(state.muted));
    renderMuteIcon();
    playTone("click");
});

const cursorDot = document.getElementById("cursorDot");
const cursorRing = document.getElementById("cursorRing");
window.addEventListener("mousemove", (event) => {
    if (!cursorDot || !cursorRing) return;
    cursorDot.style.left = `${event.clientX}px`;
    cursorDot.style.top = `${event.clientY}px`;
    cursorRing.style.left = `${event.clientX}px`;
    cursorRing.style.top = `${event.clientY}px`;
});

document.querySelectorAll("a, button, .interactive, .symptom-item").forEach((item) => {
    item.addEventListener("mouseenter", () => cursorRing?.classList.add("grow"));
    item.addEventListener("mouseleave", () => cursorRing?.classList.remove("grow"));
    item.addEventListener("click", () => playTone("click"));
});

window.addEventListener("mousemove", (event) => {
    document.querySelectorAll(".medical-bg span").forEach((item, index) => {
        const depth = (index + 1) * 0.006;
        item.style.transform = `translate(${event.clientX * depth}px, ${event.clientY * depth}px)`;
    });
});

const symptomSearch = document.getElementById("symptomSearch");
const symptomGrid = document.getElementById("symptomGrid");
if (symptomSearch && symptomGrid) {
    symptomSearch.addEventListener("input", () => {
        const query = symptomSearch.value.trim().toLowerCase();
        symptomGrid.querySelectorAll(".symptom-item").forEach((item) => {
            item.style.display = item.dataset.name.includes(query) ? "grid" : "none";
        });
    });
}

document.getElementById("clearSymptoms")?.addEventListener("click", () => {
    document.querySelectorAll(".symptom-checkbox").forEach((checkbox) => {
        checkbox.checked = false;
    });
});

document.getElementById("predictForm")?.addEventListener("submit", () => {
    document.getElementById("predictPulse")?.classList.remove("d-none");
});

document.getElementById("ajaxPredict")?.addEventListener("click", async () => {
    const resultBox = document.getElementById("ajaxResult");
    const pulse = document.getElementById("predictPulse");
    const symptoms = [...document.querySelectorAll(".symptom-checkbox:checked")].map((item) => item.value);
    pulse?.classList.remove("d-none");
    resultBox.innerHTML = '<div class="medical-loader compact"><span></span><p>Tahmin hesaplanıyor...</p></div>';

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({symptoms}),
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || "Tahmin başarısız.");
        }
        playTone("success");
        resultBox.innerHTML = renderPrediction(data);
    } catch (error) {
        playTone("error");
        resultBox.innerHTML = `<div class="alert alert-danger">${error.message}</div>`;
    } finally {
        pulse?.classList.add("d-none");
    }
});

function renderPrediction(data) {
    const alternatives = (data.alternatifler || [])
        .slice(0, 3)
        .map((item) => `<div class="alt-row"><span>${item.hastalik}</span><strong>${item.olasilik}%</strong></div>`)
        .join("");
    return `
        <div class="prediction-mini">
            <span class="badge text-bg-primary mb-2">Tahmin</span>
            <h3 class="h4 fw-bold">${data.hastalik}</h3>
            <p class="mb-2">Olasılık: <strong>${data.olasilik}%</strong></p>
            <p class="small text-secondary mb-2">Confidence ${data.confidence} · Support ${data.support} · Lift ${data.lift}</p>
            <div class="referral-message ${data.yonlendirme?.acil ? "urgent" : ""}">
                <strong>YÖNLENDİRME MESAJI:</strong> ${data.yonlendirme?.mesaj || ""}
            </div>
            <div class="mt-3">${alternatives}</div>
        </div>`;
}

async function loadPharmacies() {
    const carousel = document.getElementById("pharmacyCarousel");
    if (!carousel) return;
    carousel.innerHTML = '<div class="medical-loader compact"><span></span><p>Eczaneler yükleniyor...</p></div>';
    try {
        const response = await fetch("/pharmacies");
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Eczane bilgisi alınamadı.");
        const items = data.pharmacies || [];
        if (!items.length) {
            carousel.innerHTML = '<div class="alert alert-warning">Nöbetçi eczane kaydı bulunamadı.</div>';
            return;
        }
        carousel.innerHTML = items.map(renderPharmacy).join("");
    } catch (error) {
        carousel.innerHTML = `<div class="alert alert-danger">${error.message}</div>`;
    }
}

function renderPharmacy(item) {
    return `
        <article class="pharmacy-card">
            <span class="badge text-bg-success mb-2">${item.district || "Kars"}</span>
            <h3>${item.name}</h3>
            <p class="small text-secondary">${item.address || ""}</p>
            <a class="btn btn-sm btn-outline-primary interactive" href="tel:${(item.phone || "").replaceAll(" ", "")}">
                <i class="bi bi-telephone me-1"></i>${item.phone || "Telefon yok"}
            </a>
        </article>`;
}

document.getElementById("refreshPharmacies")?.addEventListener("click", loadPharmacies);
loadPharmacies();

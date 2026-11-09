const fs = require('fs');
const path = require('path');

const html = fs.readFileSync(path.join(__dirname, '../app/index.html'), 'utf8');

// Build a mock DOM based on index.html
const elements = new Map();

// Helper to create mock element
function getOrCreateElement(id, tag = 'div', classes = []) {
  if (elements.has(id)) return elements.get(id);
  const el = {
    id: id,
    tagName: tag.toUpperCase(),
    classList: {
      _classes: new Set(classes),
      add(c) { this._classes.add(c); },
      remove(c) { this._classes.delete(c); },
      contains(c) { return this._classes.has(c); },
      toggle(c) { if (this._classes.has(c)) this._classes.delete(c); else this._classes.add(c); }
    },
    style: {},
    attributes: new Map(),
    children: [],
    listeners: new Map(),
    value: '',
    textContent: '',
    innerHTML: '',
    disabled: false,
    setAttribute(k, v) { this.attributes.set(k, v); },
    getAttribute(k) { return this.attributes.get(k) || null; },
    addEventListener(evt, fn) {
      if (!this.listeners.has(evt)) this.listeners.set(evt, []);
      this.listeners.get(evt).push(fn);
    },
    dispatchEvent(event) {
      const fns = this.listeners.get(event.type) || [];
      fns.forEach(fn => fn(event));
    },
    appendChild(child) { this.children.push(child); return child; },
    querySelector(sel) { return null; },
    querySelectorAll(sel) { return []; },
    reset() {},
    click() {
      const fns = this.listeners.get('click') || [];
      console.log(`[CLICK] #${this.id} triggered ${fns.length} listener(s)`);
      fns.forEach(fn => fn({ target: this }));
    }
  };
  elements.set(id, el);
  return el;
}

// Parse IDs from HTML
const idRegex = /id=["']([^"']+)["']/g;
let match;
while ((match = idRegex.exec(html)) !== null) {
  getOrCreateElement(match[1]);
}

// Special attributes/values
const speedRegex = /class=["'][^"']*speed-pill[^"']*["'][^>]*data-speed=["']([^"']+)["']/g;
const speedPills = [];
while ((match = speedRegex.exec(html)) !== null) {
  const pill = getOrCreateElement(`speed-pill-${match[1]}`);
  pill.classList.add('speed-pill');
  pill.setAttribute('data-speed', match[1]);
  speedPills.push(pill);
}

const docListeners = new Map();

global.document = {
  readyState: 'loading',
  getElementById(id) {
    if (!elements.has(id)) {
      console.warn(`[DOM WARNING] getElementById NOT FOUND: "${id}"`);
      return null;
    }
    return elements.get(id);
  },
  querySelector(sel) {
    if (sel.startsWith('#')) return this.getElementById(sel.slice(1));
    if (sel.startsWith('.')) {
      const c = sel.slice(1);
      for (const el of elements.values()) {
        if (el.classList.contains(c)) return el;
      }
    }
    return getOrCreateElement(`mock-${sel}`);
  },
  querySelectorAll(sel) {
    if (sel.includes('.speed-pill')) return speedPills;
    const res = [];
    if (sel.startsWith('.')) {
      const c = sel.slice(1);
      for (const el of elements.values()) {
        if (el.classList.contains(c)) res.push(el);
      }
    }
    return res;
  },
  createElement(tag) {
    return getOrCreateElement(`created-${Math.random().toString(36).substr(2, 9)}`, tag);
  },
  addEventListener(evt, fn) {
    if (!docListeners.has(evt)) docListeners.set(evt, []);
    docListeners.get(evt).push(fn);
  }
};

global.window = global;
global.localStorage = {
  store: new Map(),
  getItem(k) { return this.store.get(k) || null; },
  setItem(k, v) { this.store.set(k, v); },
  removeItem(k) { this.store.delete(k); },
  clear() { this.store.clear(); }
};

global.Event = function(type) { this.type = type; };
global.fetch = async () => ({ ok: false, json: async () => ({}) });

console.log("Loading app.js into environment...");
try {
  require('../app/app.js');
  console.log("app.js loaded successfully without top-level syntax/eval error.");
} catch (err) {
  console.error("ERROR loading app.js:", err);
  process.exit(1);
}

console.log("Triggering DOMContentLoaded...");
const domLoadedFns = docListeners.get('DOMContentLoaded') || [];
console.log(`Found ${domLoadedFns.length} DOMContentLoaded listener(s).`);
try {
  domLoadedFns.forEach(fn => fn());
  console.log("DOMContentLoaded executed cleanly.");
} catch (err) {
  console.error("ERROR in DOMContentLoaded:", err);
}

console.log("\nTesting Start Simulation button click...");
const btnSimStart = document.getElementById('btn-sim-start');
if (!btnSimStart) {
  console.error("btn-sim-start not found!");
} else {
  console.log(`btn-sim-start listeners: ${btnSimStart.listeners.get('click')?.length || 0}`);
  try {
    btnSimStart.click();
    console.log("Click executed. Checking button state:");
    console.log(`  btnSimStart.disabled: ${btnSimStart.disabled}`);
    
    setTimeout(() => {
      console.log(`  sim-clock-display after 650ms: ${document.getElementById('sim-clock-display')?.textContent}`);
      console.log(`  op-sim-clock after 650ms: ${document.getElementById('op-sim-clock')?.textContent}`);
      process.exit(0);
    }, 650);
  } catch (err) {
    console.error("ERROR on clicking btn-sim-start:", err);
    process.exit(1);
  }
}

"""
Shared styling for the generated HTML pages (light and dark themes, tables, charts, tooltips).
"""

CSS = """
:root{color-scheme:light;
 --page:#f9f9f7;--surface:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink-2:#52514e;--muted:#898781;
 --grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;
 --good:#006300;--bad:#b42f2f;--accent-wash:#eef4fc;--warn-wash:#fdf3ea;--chip:#f0efec}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
 --page:#0d0d0d;--surface:#1a1a19;--card:#151514;--ink:#f0efec;--ink-2:#c3c2b7;--muted:#898781;
 --grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;
 --good:#0ca30c;--bad:#e66767;--accent-wash:#14202f;--warn-wash:#2a1d14;--chip:#262624}}
:root[data-theme="dark"]{color-scheme:dark;
 --page:#0d0d0d;--surface:#1a1a19;--card:#151514;--ink:#f0efec;--ink-2:#c3c2b7;--muted:#898781;
 --grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;
 --good:#0ca30c;--bad:#e66767;--accent-wash:#14202f;--warn-wash:#2a1d14;--chip:#262624}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--page);color:var(--ink);font:16px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:0 16px}
header.hero{padding:56px 0 28px}
.eyebrow{font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 10px}
h1{font-size:clamp(28px,4.4vw,42px);line-height:1.15;margin:0 0 14px;letter-spacing:-.01em;font-weight:650}
.lede{font-size:18px;color:var(--ink-2);max-width:760px;margin:0}
nav.toc{display:flex;flex-wrap:wrap;gap:8px;margin:26px 0 0}
nav.toc a{font-size:14px;color:var(--ink-2);text-decoration:none;padding:5px 12px;border-radius:999px;background:var(--chip)}
nav.toc a:hover{color:var(--ink)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:34px 0 8px}
.tile{background:var(--card);border-radius:14px;padding:18px 18px 16px;box-shadow:0 0 0 1px var(--ring)}
.tile .label{font-size:13px;color:var(--ink-2)}
.tile .value{font-size:32px;font-weight:620;margin:4px 0 2px;letter-spacing:-.01em}
.tile .sub{font-size:13px;color:var(--muted)}
section{padding:44px 0 8px}
h2{font-size:26px;margin:0 0 6px;letter-spacing:-.005em;font-weight:640}
h2 .num{color:var(--muted);font-weight:500;margin-right:8px}
h3{font-size:18px;margin:34px 0 8px;font-weight:620}
p{max-width:780px}
.intro{color:var(--ink-2);margin-top:0}
.card{background:var(--card);border-radius:16px;padding:20px 20px 14px;box-shadow:0 0 0 1px var(--ring);margin:16px 0}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px;color:var(--ink-2);margin:0 0 8px}
.key{display:inline-flex;align-items:center;gap:7px}
.key i{display:inline-block;width:18px;height:3px;border-radius:2px}
.key i.dot{width:10px;height:10px;border-radius:50%}
.key i.dia{width:9px;height:9px;transform:rotate(45deg);border:2px solid var(--ink);border-radius:1px;background:transparent}
.key i.tick{width:2px;height:14px;background:var(--ink-2)}
.key i.sq{width:12px;height:12px;border-radius:3px}
svg.chart{width:100%;height:auto;display:block;overflow:visible}
.chart .grid{stroke:var(--grid);stroke-width:1}
.chart .sep{stroke:var(--axis);stroke-width:1}
.chart .tick{fill:var(--muted);font-size:12px;font-variant-numeric:tabular-nums}
.chart .axis-title,.chart .col-head{fill:var(--muted);font-size:12px}
.chart .cat{fill:var(--ink);font-size:13px}
.chart .delta{fill:var(--ink-2);font-size:13px;font-variant-numeric:tabular-nums}
.chart .delta.neg{fill:var(--bad)}
.chart .ref{stroke:var(--muted);stroke-width:1.5;stroke-dasharray:4 4}
.chart .ref-label{fill:var(--muted);font-size:12px}
.chart .line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.chart .line.s1{stroke:var(--s1)}.chart .line.s2{stroke:var(--s2)}.chart .line.s3{stroke:var(--s3)}
.chart .dot{stroke:var(--card);stroke-width:2}
.chart .dot.s1{fill:var(--s1)}.chart .dot.s2{fill:var(--s2)}.chart .dot.s3{fill:var(--s3)}
.chart .paper-mark{fill:none;stroke:var(--ink);stroke-width:1.6}
.chart .bar.s1{fill:var(--s1)}.chart .bar.s2{fill:var(--s2)}
.chart .rand{stroke:var(--ink-2);stroke-width:2;stroke-linecap:round}
.chart .hit{fill:transparent;cursor:default}
.chart .hit:hover{fill:rgba(127,127,127,.10)}
.panels{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.panel h4{margin:0 0 4px;font-size:15px;font-weight:620}
.panel .note{font-size:13px;color:var(--muted);margin:0 0 6px}
@media (max-width:820px){.panels{grid-template-columns:1fr}}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:12px 0}
table.data{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}
table.data th,table.data td{padding:8px 10px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--grid)}
table.data thead th{font-size:12px;font-weight:600;color:var(--ink-2);border-bottom:1px solid var(--axis);vertical-align:bottom}
table.data tr.super th{border-bottom:none;text-align:center;color:var(--muted);font-weight:500;padding-bottom:0}
table.data tbody th,table.data thead tr:last-child th:first-child{text-align:left}
table.data tbody th{font-weight:500}
table.data tr.total th,table.data tr.total td{font-weight:650;border-top:1px solid var(--axis);border-bottom:none}
table.data .grp{border-left:1px solid var(--axis)}
table.data td.neg{color:var(--bad)}
table.data td.below{text-decoration:underline dotted;text-underline-offset:3px;color:var(--bad)}
table.data td.p{color:var(--ink-2)}
table.data td.muted{color:var(--muted)}
table.data .paper{display:block;font-size:11px;color:var(--muted);line-height:1.2}
table.data.compact td,table.data.compact th{padding:6px 10px}
table.data td.names{white-space:normal;text-align:left;min-width:420px;color:var(--ink-2)}
table.data td.nowrap{white-space:nowrap}
.caption{font-size:13px;color:var(--muted);margin:6px 0 0;max-width:none}
.heats{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
figure.heat{margin:0}
figure.heat figcaption{font-size:14px;font-weight:600;margin:0 0 8px}
figure.heat figcaption small{display:block;font-weight:400;color:var(--muted)}
table.heatmap{border-collapse:separate;border-spacing:2px;font-size:12.5px;font-variant-numeric:tabular-nums;width:100%}
table.heatmap td{padding:7px 0;text-align:center;border-radius:4px}
table.heatmap th{font-size:12px;font-weight:500;color:var(--muted);padding:2px 4px}
.callouts{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px;margin-top:14px}
.callout{background:var(--card);border-radius:14px;padding:18px;box-shadow:0 0 0 1px var(--ring)}
.callout h4{margin:0 0 6px;font-size:15px}
.callout p{margin:0;font-size:14.5px;color:var(--ink-2)}
.callout .tag{display:inline-block;font-size:11px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;padding:2px 8px;border-radius:999px;background:var(--chip);color:var(--ink-2);margin-bottom:8px}
.highlights{list-style:none;padding:0;margin:16px 0;display:grid;gap:10px;max-width:820px}
.highlights li{padding:12px 16px 12px 16px;background:var(--accent-wash);border-radius:12px;font-size:15px}
.highlights li.warn{background:var(--warn-wash)}
details{margin:10px 0}
summary{cursor:pointer;color:var(--ink-2);font-size:14px}
img.figure{width:100%;height:auto;border-radius:12px;background:#fff;display:block;margin-top:10px}
code{font:13px/1.4 ui-monospace,SFMono-Regular,Consolas,monospace;background:var(--chip);padding:1px 6px;border-radius:6px}
pre{background:var(--card);box-shadow:0 0 0 1px var(--ring);border-radius:12px;padding:14px 16px;overflow-x:auto;font:13px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace}
.status{display:inline-flex;align-items:center;gap:6px;font-size:13px;font-weight:600}
.status.ok{color:var(--good)}.status.no{color:var(--bad)}.status.part{color:var(--ink-2)}
footer{padding:48px 0 64px;color:var(--muted);font-size:13px}
#tip{position:fixed;pointer-events:none;z-index:10;background:var(--card);color:var(--ink);box-shadow:0 0 0 1px var(--ring),0 6px 24px rgba(0,0,0,.12);
 border-radius:10px;padding:8px 11px;font-size:13px;line-height:1.45;opacity:0;transition:opacity .12s;max-width:260px}
"""

JS = """
const tip=document.getElementById('tip');
function show(el,e){tip.innerHTML=el.dataset.tip;tip.style.opacity=1;
  const x=Math.min(e.clientX+14,window.innerWidth-tip.offsetWidth-8);tip.style.left=x+'px';tip.style.top=(e.clientY+14)+'px';}
document.querySelectorAll('[data-tip]').forEach(el=>{
  el.addEventListener('mouseenter',e=>show(el,e));
  el.addEventListener('mousemove',e=>show(el,e));
  el.addEventListener('mouseleave',()=>{tip.style.opacity=0;});
});
"""

"""Cheap local HTML-link, grid and summary validation. Does not train models."""
import ast
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import runpy

ROOT=Path(__file__).resolve().parents[1]

class Page(HTMLParser):
    def __init__(self,text):
        super().__init__(); self.ids=set(); self.links=[]; self.feed(text)
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if 'id' in attrs: self.ids.add(attrs['id'])
        for key in ['href','src']:
            if key in attrs: self.links.append(attrs[key])

pages={p:Page(p.read_text(encoding='utf-8')) for p in ROOT.glob('*.html')}
for path,page in pages.items():
    for link in page.links:
        url=urlsplit(link)
        if url.scheme or url.netloc: continue
        target=(path.parent/unquote(url.path)).resolve() if url.path else path
        assert target.exists(), (path.name,link)
        if url.fragment and target in pages:
            assert unquote(url.fragment) in pages[target].ids, (path.name,link)
for path in (ROOT/'code').glob('*.py'):
    ast.parse(path.read_text(encoding='utf-8'))
runner=runpy.run_path(str(ROOT/'code/run_grid.py'))
grid=runner['grid']()
assert len(grid)==24 and len({tuple(x.items()) for x in grid})==24
assert {round(x['b']) for x in grid}=={9,19}
runpy.run_path(str(ROOT/'code/verify_summary.py'),run_name='__main__')
print('HTML local targets/anchors, Python syntax, 24-point grid: PASS')

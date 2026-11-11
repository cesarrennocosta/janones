import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from types import SimpleNamespace

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# Enable single text object emission by zeroing prev_kern
from matplotlib import _text_helpers
orig_layout = _text_helpers.layout
def nokern_layout(*args, **kwargs):
    for item in orig_layout(*args, **kwargs):
        yield SimpleNamespace(
            ft_object=item.ft_object,
            char=item.char,
            glyph_index=item.glyph_index,
            prev_kern=0,
            x=item.x,
            y=item.y
        )
_text_helpers.layout = nokern_layout

print("Illustrator patch applied successfully!")

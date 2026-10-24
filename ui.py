"""
Screen-space UI toolkit for the tracker: theme, text, panels, chips.

Everything here is drawn as VTK 2D actors in pixel coordinates. Panels size themselves
to their content and are re-laid out every frame, so they follow window resizes.
"""

import os

import pyvista as pv
from vtkmodules.vtkCommonCore import VTK_FONT_FILE, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkRenderingCore import vtkActor2D, vtkPolyDataMapper2D, vtkTextActor


# -----------------------------------------------------------------------------
# Theme
# -----------------------------------------------------------------------------
BACKGROUND = "#03050a"
PANEL_COLOR = "#0b111c"
PANEL_OPACITY = 0.86
BORDER_COLOR = "#1c2738"
HOVER_COLOR = "#17233a"
ACCENT = "#38bdf8"
TEXT = "#e5ecf5"
MUTED = "#7d8ba1"
DISABLED = "#3e4a5c"
WARNING = "#fbbf24"
CHIP_COLOR = "#152033"
CHIP_HOVER_COLOR = "#1f2e47"


def _font(*names):
    font_dir = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    for name in names:
        path = os.path.join(font_dir, name)
        if os.path.exists(path):
            return path
    return None


FONT_REGULAR = _font("segoeui.ttf")
FONT_SEMIBOLD = _font("seguisb.ttf", "segoeuib.ttf")
FONT_MONO = _font("consola.ttf")
FONT_SYMBOL = _font("seguisym.ttf")


# -----------------------------------------------------------------------------
# Text
# -----------------------------------------------------------------------------
_line_height_cache = {}


def make_text(renderer, size, color=TEXT, font=FONT_REGULAR, justify="left", text=""):
    actor = vtkTextActor()
    prop = actor.GetTextProperty()
    prop.SetFontSize(size)
    prop.SetColor(pv.Color(color).float_rgb)
    if font:
        prop.SetFontFamily(VTK_FONT_FILE)
        prop.SetFontFile(font)
    else:
        prop.SetFontFamilyToArial()
    {"left": prop.SetJustificationToLeft,
     "center": prop.SetJustificationToCentered,
     "right": prop.SetJustificationToRight}[justify]()
    prop.SetVerticalJustificationToBottom()
    actor.SetInput(text)
    actor.SetPickable(False)
    renderer.AddViewProp(actor)
    return actor


def text_size(renderer, actor):
    """Width of the actual text, height from the font metrics so rows line up."""
    size = [0.0, 0.0]
    actor.GetSize(renderer, size)

    prop = actor.GetTextProperty()
    key = (prop.GetFontFile(), prop.GetFontSize())
    if key not in _line_height_cache:
        _line_height_cache[key] = size[1]
    return size[0], _line_height_cache[key]


def set_text_color(actor, color):
    actor.GetTextProperty().SetColor(pv.Color(color).float_rgb)


# -----------------------------------------------------------------------------
# Shapes
# -----------------------------------------------------------------------------
class Rect:
    """A filled (or outlined) screen-space rectangle.

    Keep fills that sit underneath text opaque: VTK draws translucent 2D shapes after
    all text, so a translucent fill would cover text added before it.
    """

    def __init__(self, renderer, color, opacity=1.0, outline=False):
        self.points = vtkPoints()
        self.points.SetNumberOfPoints(4)

        cells = vtkCellArray()
        ids = (0, 1, 2, 3, 0) if outline else (0, 1, 2, 3)
        cells.InsertNextCell(len(ids))
        for i in ids:
            cells.InsertCellPoint(i)

        poly = vtkPolyData()
        poly.SetPoints(self.points)
        if outline:
            poly.SetLines(cells)
        else:
            poly.SetPolys(cells)

        mapper = vtkPolyDataMapper2D()
        mapper.SetInputData(poly)

        self.actor = vtkActor2D()
        self.actor.SetMapper(mapper)
        self.actor.SetPickable(False)
        self.set_color(color, opacity)
        self.place(0, 0, 0, 0)
        renderer.AddViewProp(self.actor)

    def set_color(self, color, opacity=None):
        prop = self.actor.GetProperty()
        prop.SetColor(pv.Color(color).float_rgb)
        if opacity is not None:
            prop.SetOpacity(opacity)

    def place(self, x, y, w, h):
        # Outlines sit on pixel centres so they render crisp
        for i, (px, py) in enumerate(((x, y), (x + w, y), (x + w, y + h), (x, y + h))):
            self.points.SetPoint(i, px + 0.5, py + 0.5, 0)
        self.points.Modified()

    def set_visible(self, visible):
        self.actor.SetVisibility(visible)


class Chip:
    """A small pill button."""

    def __init__(self, renderer, label, size, key):
        self.key = key
        self.rect = Rect(renderer, CHIP_COLOR)
        self.text = make_text(renderer, size, TEXT, FONT_SEMIBOLD, justify="center", text=label)
        self.box = (0, 0, 0, 0)
        self.active = False
        self.active_color = ACCENT
        self.hovered = False

    def set_active(self, active, color=ACCENT):
        self.active = active
        self.active_color = color
        self._restyle()

    def set_hovered(self, hovered):
        if hovered != self.hovered:
            self.hovered = hovered
            self._restyle()

    def _restyle(self):
        if self.active:
            self.rect.set_color(self.active_color)
            set_text_color(self.text, BACKGROUND)
        else:
            self.rect.set_color(CHIP_HOVER_COLOR if self.hovered else CHIP_COLOR)
            set_text_color(self.text, TEXT)

    def contains(self, x, y):
        x0, y0, w, h = self.box
        return x0 <= x <= x0 + w and y0 <= y <= y0 + h


# -----------------------------------------------------------------------------
# Panel
# -----------------------------------------------------------------------------
class Panel:
    """A padded card holding stacked rows: plain text, label/value fields and chip rows.

    Rows (fields) and chips created with a ``key`` are clickable: ``hit()`` returns the key
    under the mouse and ``set_hover()`` highlights it.
    """

    PAD_X = 16
    PAD_Y = 13
    ROW_GAP = 4
    COL_GAP = 28
    CHIP_PAD_X = 11
    CHIP_PAD_Y = 5
    CHIP_GAP = 6

    def __init__(self, renderer, accent=None, min_width=0, opacity=PANEL_OPACITY):
        self.renderer = renderer
        self.background = Rect(renderer, PANEL_COLOR, opacity)
        self.border = Rect(renderer, BORDER_COLOR, 1.0, outline=True)
        self.accent = Rect(renderer, accent) if accent else None
        # Created before any row so row text draws on top of it
        self.highlight = Rect(renderer, HOVER_COLOR)
        self.highlight.set_visible(False)
        self.min_width = min_width
        self.rows = []
        self.visible = True
        self.box = (0, 0, 0, 0)
        self.row_boxes = {}
        self.hover_key = None

    # ---- row builders ----------------------------------------------------------
    def add_text(self, size, color=TEXT, font=FONT_REGULAR, text=""):
        actor = make_text(self.renderer, size, color, font, text=text)
        self.rows.append({"kind": "text", "actor": actor})
        return actor

    def add_field(self, label, size=12, glyph_color=None, value_color=TEXT, value_font=FONT_MONO, key=None):
        glyph = None
        if glyph_color:
            glyph = make_text(self.renderer, size, glyph_color, FONT_SYMBOL or FONT_REGULAR, text="●")
        label_actor = make_text(self.renderer, size, TEXT if glyph or key else MUTED, FONT_REGULAR, text=label)
        value = make_text(self.renderer, size, value_color, value_font, justify="right")
        self.rows.append({"kind": "field", "glyph": glyph, "label": label_actor, "value": value, "key": key})
        return value

    def field(self, key):
        """The row dict for a keyed field (to restyle its glyph/label/value)."""
        return next(row for row in self.rows if row.get("key") == key)

    def add_gap(self, pixels):
        self.rows.append({"kind": "gap", "height": pixels})

    def add_chips(self, labels, keys=None, size=11):
        keys = keys or labels
        chips = [Chip(self.renderer, label, size, key) for label, key in zip(labels, keys)]
        self.rows.append({"kind": "chips", "chips": chips})
        return chips

    # ---- visibility / interaction -------------------------------------------
    @staticmethod
    def _row_actors(row):
        if row["kind"] == "text":
            yield row["actor"]
        elif row["kind"] == "field":
            yield from (a for a in (row["glyph"], row["label"], row["value"]) if a is not None)
        elif row["kind"] == "chips":
            for chip in row["chips"]:
                yield chip.rect.actor
                yield chip.text

    def set_visible(self, visible):
        if visible == self.visible:
            return
        self.visible = visible
        for rect in (self.background, self.border, self.accent):
            if rect:
                rect.set_visible(visible)
        for row in self.rows:
            for actor in self._row_actors(row):
                actor.SetVisibility(visible and not row.get("hidden"))
        if not visible:
            self.set_hover(None)

    def set_row_visible(self, ref, visible):
        """Show/hide one row, found by its key or by any of its text actors. Hidden rows take no space."""
        for row in self.rows:
            if ref == row.get("key") or any(ref is actor for actor in self._row_actors(row)):
                row["hidden"] = not visible
                for actor in self._row_actors(row):
                    actor.SetVisibility(self.visible and visible)
                return

    def contains(self, x, y):
        x0, y0, w, h = self.box
        return self.visible and x0 <= x <= x0 + w and y0 <= y <= y0 + h

    def chips(self):
        for row in self.rows:
            if row["kind"] == "chips":
                yield from row["chips"]

    def hit(self, x, y):
        """Key of the clickable row or chip at (x, y), or None."""
        if not self.contains(x, y):
            return None
        for chip in self.chips():
            if chip.contains(x, y):
                return chip.key
        for key, (bx, by, bw, bh) in self.row_boxes.items():
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return key
        return None

    def set_hover(self, key):
        self.hover_key = key
        for chip in self.chips():
            chip.set_hovered(chip.key == key)
        self.highlight.set_visible(key in self.row_boxes)

    # ---- layout ----------------------------------------------------------------
    def layout(self, x, y, anchor_x=0.0, anchor_y=0.0, clamp_to=None):
        """Size the panel to its content and place it so (x, y) sits at the given anchor.

        anchor_x: 0 = left edge, 0.5 = centre, 1 = right edge (same idea for anchor_y, 0 = bottom).
        """
        if not self.visible:
            return

        ren = self.renderer
        measured = []
        label_col = value_col = 0
        rows = [row for row in self.rows if not row.get("hidden")]
        for row in rows:
            kind = row["kind"]
            if kind == "text":
                w, h = text_size(ren, row["actor"])
            elif kind == "field":
                glyph_w = text_size(ren, row["glyph"])[0] + 8 if row["glyph"] else 0
                label_w, h = text_size(ren, row["label"])
                label_col = max(label_col, glyph_w + label_w)
                value_col = max(value_col, text_size(ren, row["value"])[0])
                w = 0
            elif kind == "chips":
                sizes = [text_size(ren, chip.text) for chip in row["chips"]]
                w = sum(s[0] + 2 * self.CHIP_PAD_X for s in sizes) + self.CHIP_GAP * (len(sizes) - 1)
                h = max(s[1] for s in sizes) + 2 * self.CHIP_PAD_Y
            else:
                w, h = 0, row["height"]
            measured.append((w, h))

        inner_w = max([w for w, _ in measured] + [label_col + self.COL_GAP + value_col if value_col else 0])
        width = max(inner_w + 2 * self.PAD_X, self.min_width)
        height = 2 * self.PAD_Y + sum(h for _, h in measured)
        height += self.ROW_GAP * sum(1 for row in rows if row["kind"] != "gap")

        x0 = round(x - anchor_x * width)
        y0 = round(y - anchor_y * height)
        if clamp_to:
            win_w, win_h = clamp_to
            x0 = min(max(x0, 8), win_w - width - 8)
            y0 = min(max(y0, 8), win_h - height - 8)

        self.box = (x0, y0, width, height)
        self.background.place(x0, y0, width, height)
        self.border.place(x0, y0, width, height)
        if self.accent:
            self.accent.place(x0, y0, 3, height)

        left = x0 + self.PAD_X
        right = x0 + width - self.PAD_X
        cursor = y0 + height - self.PAD_Y
        self.row_boxes = {}
        for row, (w, h) in zip(rows, measured):
            kind = row["kind"]
            cursor -= h
            if kind == "gap":
                continue
            if kind == "text":
                row["actor"].SetDisplayPosition(left, round(cursor))
            elif kind == "field":
                label_x = left
                if row["glyph"]:
                    row["glyph"].SetDisplayPosition(left, round(cursor))
                    label_x += text_size(ren, row["glyph"])[0] + 8
                row["label"].SetDisplayPosition(round(label_x), round(cursor))
                row["value"].SetDisplayPosition(round(right), round(cursor))
                if row["key"] is not None:
                    self.row_boxes[row["key"]] = (x0 + 4, cursor - 2, width - 8, h + 4)
            elif kind == "chips":
                chip_x = left
                for chip in row["chips"]:
                    cw = text_size(ren, chip.text)[0] + 2 * self.CHIP_PAD_X
                    chip.box = (chip_x, cursor, cw, h)
                    chip.rect.place(chip_x, cursor, cw, h)
                    chip.text.SetDisplayPosition(round(chip_x + cw / 2), round(cursor + self.CHIP_PAD_Y))
                    chip_x += cw + self.CHIP_GAP
            cursor -= self.ROW_GAP

        if self.hover_key in self.row_boxes:
            self.highlight.place(*self.row_boxes[self.hover_key])
            self.highlight.set_visible(True)
        else:
            self.highlight.set_visible(False)

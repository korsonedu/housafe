# -*- coding: utf-8 -*-
"""家安 housafe BP → pptx 生成器 v2 · 高密度阅读版
设计原则：信息密度最大化、布局多变、不用卡片堆砌、不出现"不是而是"句式。
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

# ================= 视觉常量 =================
IKB       = RGBColor(0x00,0x2F,0xA7)
IKB_BRT   = RGBColor(0x5B,0x7B,0xFF)
IKB_DARK  = RGBColor(0x00,0x20,0x70)
INK       = RGBColor(0x0A,0x0A,0x0A)
PAPER     = RGBColor(0xFA,0xFA,0xF8)
GREY1     = RGBColor(0xF0,0xF0,0xEE)
GREY2     = RGBColor(0xD4,0xD4,0xD2)
GREY3     = RGBColor(0x73,0x73,0x73)
WHITE     = RGBColor(0xFF,0xFF,0xFF)
INK_CARD  = RGBColor(0xF5,0xF5,0xF4)

SANS = 'Arial'
ZH   = '微软雅黑'
EMU_W, EMU_H = Inches(13.333), Inches(7.5)
M = 0.35  # 紧凑页边距

# ================= 绘制原语 =================
def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])

def bg(slide, color):
    r = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, EMU_W, EMU_H)
    r.fill.solid(); r.fill.fore_color.rgb = color
    r.line.fill.background(); r.shadow.inherit = False
    sp = r._element; sp.getparent().remove(sp); slide.shapes._spTree.insert(2, sp)
    return r

def rect(slide, x, y, w, h, fill=None, line=None, line_w=1.0):
    r = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill is None: r.fill.background()
    else: r.fill.solid(); r.fill.fore_color.rgb = fill
    if line is None: r.line.fill.background()
    else: r.line.color.rgb = line; r.line.width = Pt(line_w)
    r.shadow.inherit = False
    return r

def hline(slide, x, y, w, color, weight=1.0):
    ln = slide.shapes.add_connector(2, Inches(x), Inches(y), Inches(x+w), Inches(y))
    ln.line.color.rgb = color; ln.line.width = Pt(weight)
    ln.shadow.inherit = False
    return ln

def _ea(run, ea):
    rPr = run._r.get_or_add_rPr()
    e = rPr.find(qn('a:ea'))
    if e is None:
        e = rPr.makeelement(qn('a:ea'), {}); rPr.append(e)
    e.set('typeface', ea)

def add_run(p, text, size, color=INK, bold=False, italic=False, latin=SANS, ea=ZH, spc=None):
    r = p.add_run(); r.text = text
    f = r.font; f.size = Pt(size); f.bold = bold; f.italic = italic
    f.color.rgb = color; f.name = latin
    _ea(r, ea)
    if spc is not None:
        r._r.get_or_add_rPr().set('spc', str(int(spc*100)))
    return r

def box(slide, x, y, w, h, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tf

def put(slide, x, y, w, h, text, size, color=INK, bold=False, align=PP_ALIGN.LEFT,
        anchor=MSO_ANCHOR.TOP, latin=SANS, ea=ZH, italic=False, spc=None, lh=None):
    tf = box(slide, x, y, w, h, anchor)
    p = tf.paragraphs[0]; p.alignment = align
    if lh: p.line_spacing = lh
    add_run(p, text, size, color, bold, italic, latin, ea, spc)
    return tf

def mput(slide, x, y, w, h, lines, size, color=INK, lh=1.2, bold_first=False):
    """Multi-line text block. lines is a list of (text, bold) tuples or plain strings."""
    tf = box(slide, x, y, w, h)
    for i, line in enumerate(lines):
        if isinstance(line, str):
            txt, bd = line, False
        else:
            txt, bd = line
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = lh
        is_bold = bd or (bold_first and i == 0)
        add_run(p, txt, size, color, bold=is_bold)
    return tf

def header(slide, left, right, dark=False):
    c = WHITE if dark else GREY3
    put(slide, M, 0.18, 8, 0.25, left, 8, c, spc=1.2)
    put(slide, 7.0, 0.18, 5.98, 0.25, right, 8, c, align=PP_ALIGN.RIGHT, spc=1.2)

def kicker(slide, x, y, text, color=IKB):
    put(slide, x, y, 9, 0.22, text, 9, color, bold=True, spc=1.5)

def footnote(slide, text, dark=False, y=7.0):
    c = WHITE if dark else GREY3
    put(slide, M, y, 12.63, 0.35, text, 7.5, c, lh=1.15)

def title(slide, x, y, text, size=22, color=INK):
    tf = box(slide, x, y, 12.63, size/35+0.4)
    p = tf.paragraphs[0]; p.line_spacing = 1.05
    for i, line in enumerate(text.split('\n')):
        if i: p = tf.add_paragraph(); p.line_spacing = 1.05
        add_run(p, line, size, color, bold=False)
    return tf

# ================= 第 1 页 · 封面 =================
def p1(prs):
    s = blank(prs); bg(s, IKB)
    header(s, '家安 housafe · 商业计划书', '2026', dark=True)
    kicker(s, M, 1.2, 'HOUSAFE · 居家养老 AI 健康守护', WHITE)

    tf = box(s, M, 1.55, 12.63, 1.6)
    for i, line in enumerate(['让每一位独居老人，被持续看见']):
        p = tf.paragraphs[0] if i==0 else tf.add_paragraph()
        p.line_spacing = 1.08
        add_run(p, line, 36, WHITE, italic=(i==1))

    hline(s, M, 3.3, 12.63, RGBColor(0x6E,0x8A,0xC9), 1)

    put(s, M, 3.5, 12.63, 0.8,
        '毫米波雷达组网，不拍视频、不用穿戴——AI 感知老人在家状态，异常分级通知异地子女。'
        '从跌倒监测切入，做居家养老的健康基础设施。专为独居老人设计，夫妻同居场景提供基础报警。',
        11, WHITE, lh=1.3)

    # 底部三行关键信息
    items = [
        '硬件 299–799 元一次买断 · 高级健康服务 199 元/年',
        '天使轮 · 拟融资 800 万人民币 / 18 个月跑道 · 出让约 15–20%',
        'housafe · 2026'
    ]
    for i, item in enumerate(items):
        put(s, M + (0 if i < 2 else 0), 4.6 + i*0.32, 12.63, 0.28,
            item, 9, RGBColor(0xD5,0xDD,0xF2) if i < 2 else WHITE,
            align=PP_ALIGN.RIGHT if i == 2 else PP_ALIGN.LEFT, spc=1.1)

# ================= 第 2 页 · 机会速览（六卡但填充更密）=================
def p2(prs):
    s = blank(prs); bg(s, PAPER)
    header(s, 'Executive Summary · 机会速览', '02')
    kicker(s, M, 0.58, 'THE OPPORTUNITY')
    title(s, M, 0.78, '一页看懂家安 housafe', 20)

    cards = [
        ('01 痛点', '2.2 亿独居/空巢老人的跌倒检测盲区',
         '跌倒是 65+ 伤害死亡首位原因。约 30% 每年至少跌倒一次、年逾 4000 万人。47% 摔倒后无法自行起身，20–30% 倒地超 1 小时。异地子女无从知晓。'),
        ('02 市场', '银发经济 7 万亿 → 30 万亿（2035E）',
         '智能养老设备 2025 约 1840 亿、年增约 20%。健康监测可穿戴 25H1 出货 +53.6%。国办发〔2024〕1 号明确写入"家庭配备智能安全监护设备"。'),
        ('03 产品', '全屋毫米波雷达组网，无感守护',
         '2–4 个雷达组网，Φ85mm 像烟雾探测器，15 分钟自装。不穿戴、不操作、无摄像头。AI 识别跌倒/静止/体征异常/模式偏离四类事件，分级通知子女。'),
        ('04 壁垒', 'ELPD 统一隐空间世界模型 + 数据飞轮',
         '自研 Encoder 从原始点云提取全维度信号，云端 GPU 深度时序模型，自监督持续进化。多雷达聚合全屋画像，14 天个人基线，越用越准。'),
        ('05 模式', '硬件买断获客 + 订阅服务盈利',
         '硬件 299–799 元一次买断（微利），高级健康服务 199 元/年（高毛利，约 90%）。早期 LTV/CAC=0.77，国产 SoC 降本后转正到 2.19。'),
        ('06 融资', '天使轮 800 万 · 18 个月跑道',
         '覆盖 MVP 验证到国产 SoC 降本样机。达成数千户商用与数据飞轮，为 A 轮铺路。出让约 15–20%，投前估值约 3200–4500 万。'),
    ]
    cw, ch = 4.1, 1.95
    gx, gy = 0.15, 0.12
    x0, y0 = M, 1.45
    for i, (no, t, d) in enumerate(cards):
        r, c = divmod(i, 3)
        x, y = x0 + c*(cw+gx), y0 + r*(ch+gy)
        accent = (i == 5)
        fill = IKB if accent else INK_CARD
        rect(s, x, y, cw, ch, fill=fill)
        tc = WHITE if accent else GREY3
        tb = WHITE if accent else INK
        td = RGBColor(0xEC,0xF0,0xFB) if accent else RGBColor(0x33,0x33,0x33)
        put(s, x+0.15, y+0.1, cw-0.25, 0.22, no, 8, tc, bold=True, spc=1.0)
        put(s, x+0.15, y+0.32, cw-0.3, 0.38, t, 11, tb, lh=1.15)
        put(s, x+0.15, y+0.72, cw-0.3, ch-0.85, d, 8, td, lh=1.22)

    put(s, M, 6.92, 12.63, 0.28,
        '数据来源：民政部《2024 国家老龄事业发展公报》· CDC《老年人跌倒干预技术指南》· 《银发经济蓝皮书(2024)》· 中商产业研究院(2025) · IDC(2025H1)',
        7, GREY3)

# ================= 第 3 页 · 痛点 =================
def p3(prs):
    s = blank(prs); bg(s, INK)
    header(s, 'The Problem · 需求痛点', '03', dark=True)
    kicker(s, M, 0.9, '被忽视的高频高危', WHITE)

    # 左：大字
    tf = box(s, M, 1.3, 5.0, 2.0)
    for i, line in enumerate(['每 3 位老人', '就有 1 人', '每年跌倒']):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = 1.05
        add_run(p, line, 30, WHITE, italic=(i == 2))

    put(s, M, 3.5, 5.0, 1.2,
        '2.2 亿 65+ 老人（占 15.6%，2035 年约 3 亿）。90% 居家养老。'
        '独居/空巢约 1.2 亿——父母是否安好，异地子女无从知晓。'
        '跌倒是 65+ 伤害死亡首位、创伤性骨折首位。独居老人倒下后，抢救黄金期在无人知晓中流逝。',
        9, RGBColor(0xC8,0xC8,0xC8), lh=1.25)

    # 右：数据 + 方案对比
    rx = 6.0
    hline(s, rx, 1.3, 6.98, RGBColor(0x55,0x55,0x55), 1)

    # 数据块
    put(s, rx, 1.4, 6.98, 0.2, '年跌倒人数', 8, GREY3, spc=1.0)
    put(s, rx, 1.58, 6.98, 0.55, '4000万+', 26, WHITE)
    put(s, rx, 2.15, 6.98, 0.5,
        '约 30% 每年至少跌倒一次。跌倒是伤害死亡首位、创伤性骨折首位。',
        9, RGBColor(0xC8,0xC8,0xC8), lh=1.2)

    hline(s, rx, 2.75, 6.98, RGBColor(0x55,0x55,0x55), 1)
    put(s, rx, 2.85, 6.98, 0.2, '摔倒后无法自行起身', 8, GREY3, spc=1.0)
    put(s, rx, 3.03, 6.98, 0.55, '47%', 26, WHITE)
    put(s, rx, 3.6, 6.98, 0.5,
        '20–30% 倒地超 1 小时。髋部骨折 1 年死亡率 14–25%，人均住院约 1.98 万元。',
        9, RGBColor(0xC8,0xC8,0xC8), lh=1.2)

    hline(s, rx, 4.2, 6.98, RGBColor(0x55,0x55,0x55), 1)
    put(s, rx, 4.3, 6.98, 0.2, '现有方案为什么不行', 8, GREY3, spc=1.0)
    fails = [
        '摄像头 — 浴室卧室不可装，隐私争议大，老人抵触强',
        '手环/手表 — 依从性仅 30–50%，忘戴、洗澡取下、充电时失效',
        '紧急按钮 — 摔倒后够不到，意识丧失时完全无效',
        '现有雷达 — 单房间覆盖、误报率高、精度出厂定型、无长期趋势',
    ]
    for i, f in enumerate(fails):
        put(s, rx, 4.58 + i*0.32, 6.98, 0.28, f, 8.5, RGBColor(0xBB,0xBB,0xBB), lh=1.15)

    footnote(s, '来源：民政部《2024 国家老龄事业发展公报》· CDC《老年人跌倒干预技术指南》(2011)/《核心信息》(2021) · BMJ Long-lie 研究(2023)',
             dark=True, y=7.1)

# ================= 第 4 页 · 市场 =================
def p4(prs):
    s = blank(prs); bg(s, PAPER)
    header(s, 'The Market · 市场规模', '04')
    kicker(s, M, 0.58, 'WHY NOW')
    title(s, M, 0.78, '一个正在爆发的银发健康市场', 20)

    # 说明段
    put(s, M, 1.35, 12.63, 0.42,
        '银发经济 2024 约 7 万亿 → 2035 约 30 万亿（CAGR≈15%）。我们切入其中一块具体刚需：独居老人室内无感监测。'
        '需求、供给、政策三端同时在 2024–2025 年到位。',
        9.5, RGBColor(0x33,0x33,0x33), lh=1.25)

    # 四塔
    base = 4.6; cw = 3.05; gx = 0.12; x0 = M
    towers = [
        ('TAM · 银发经济 2035E', '30 万亿', '2024 约 7 万亿，CAGR≈15%\n占 GDP 6%→10%', True),
        ('SAM · 智能养老设备', '≈2400 亿', '设备 1840 亿 + 适老化\n≈600 亿（2025），年增≈20%', False),
        ('刚需人群 · 年跌倒人次', '4000 万+', '65+ 老人跌倒/年\n居家场景最刚性需求', False),
        ('需求信号 · 出货增速', '+53.6%', '健康监测可穿戴 25H1 出货\n市场用钱投票', False),
    ]
    heights = [3.0, 1.8, 2.2, 1.3]
    for i, ((lbl, num, sub, acc), h) in enumerate(zip(towers, heights)):
        x = x0 + i*(cw+gx)
        y = base - h
        rect(s, x, y, cw, h, fill=(IKB if acc else PAPER), line=(None if acc else GREY2), line_w=1)
        nc = WHITE if acc else INK
        sc = RGBColor(0xDD,0xE4,0xF5) if acc else GREY3
        lc = WHITE if acc else GREY3
        put(s, x+0.12, y+0.1, cw-0.2, 0.2, lbl, 7.5, lc, spc=1.0)
        put(s, x+0.12, y+0.35, cw-0.2, 0.55, num, 21, nc)
        put(s, x+0.12, y+h-0.55, cw-0.2, 0.5, sub, 7.5, sc, lh=1.15)

    # Why Now 四力
    hline(s, M, 4.85, 12.63, GREY2, 1)
    put(s, M, 4.95, 12.63, 0.2, 'Why Now — 四个同时到位的驱动力', 8.5, IKB, bold=True, spc=1.0)
    forces = [
        ('① 雷达降本：60GHz SoC 从千元级走向百元内，消费级定价成为可能。'),
        ('② AI 成熟：云端深度模型处理稀疏点云可行，GPU 推理成本降至可接受范围。'),
        ('③ 老龄化拐点：65+ 已 2.2 亿（15.6%），深度老龄化社会，独居/空巢超半数。'),
        ('④ 政策明确：国办发〔2024〕1 号写入"家庭配备智能安全监护设备"。毫米波雷达已入工信部/民政部/卫健委《推广目录》。'),
    ]
    for i, f in enumerate(forces):
        put(s, M + (i%2)*6.35, 5.22 + (i//2)*0.42, 6.2, 0.38, f, 8.5, RGBColor(0x33,0x33,0x33), lh=1.2)

    footnote(s, '来源：《银发经济蓝皮书(2024)》社科文献出版社 · 中商产业研究院(2025) · IDC(2025H1) · 国办发〔2024〕1 号。市场金额为机构测算。')

# ================= 第 5 页 · 产品（布局：左侧4步流程 + 右侧场景/规格）=================
def p5(prs):
    s = blank(prs); bg(s, PAPER)
    header(s, 'The Product · 产品', '05')
    kicker(s, M, 0.58, 'HOW IT WORKS')
    title(s, M, 0.78, '全屋雷达组网，无感守护', 20)

    put(s, M, 1.3, 12.63, 0.5,
        '子女带回家，一人操作、15 分钟装完、无需工具。雷达 7×24 采集，云端 AI 识别异常，异地子女第一时间知道。'
        '老人不穿戴、不操作——对外只说是"自动感应开灯的"。系统检测到长期两人同住时自动降级为基础报警模式。',
        9.5, RGBColor(0x33,0x33,0x33), lh=1.25)

    # 左：4 步流程
    steps = [
        ('01 采集', '毫米波雷达组网',
         'Φ85mm 白色感应器，小于烟雾探测器。天花板/墙壁 3M 磁吸安装，2.4G WiFi 直连云，USB-C 供电 < 2W，全年电费 < 40 元。采集姿态、活动、呼吸、心率。安装配件包含底座、线卡、定位卡纸。'),
        ('02 感知', '云端 AI 点云理解',
         '原始点云通过 WiFi 直传云端，GPU 自研模型从点云中同时提取姿态序列、速度分布、点云形变、微多普勒谱、生命体征、活动片段六类信号。识别跌倒/长时间静止/体征异常/模式偏离四类事件。'),
        ('03 判决', '延迟确认降误报',
         '跌倒初判后等 3–5 秒观察后续行为，多雷达交叉验证，结合 14 天个人基线。区分真摔与快速坐下/蹲下/弯腰捡物——这是厂商固件做不到的，因为固件没有个人的"正常"做参照。'),
        ('04 通知', '分级触达子女',
         '不提 / App 推送 / 强推+短信 / 电话四级。紧急事件（跌倒）同时短信+电话。日常模式偏离（活动量连续下降、睡眠恶化）以周报形式呈现，不实时骚扰。'),
    ]
    for i, (no, t, d) in enumerate(steps):
        y = 1.95 + i*1.28
        hline(s, M, y, 7.8, IKB, 1.5)
        put(s, M, y+0.05, 1.5, 0.2, no, 8.5, GREY3, spc=0.8)
        put(s, M+1.6, y+0.05, 6.2, 0.26, t, 11, INK, bold=True)
        put(s, M+1.6, y+0.32, 6.2, 0.85, d, 8, RGBColor(0x33,0x33,0x33), lh=1.18)

    # 右：场景 + 硬边界
    rx = 8.55; rw = 4.43
    rect(s, rx, 1.95, rw, 3.28, fill=INK_CARD)
    put(s, rx+0.12, 2.05, rw-0.24, 0.2, '六大核心场景', 9, IKB, bold=True, spc=1.0)
    scenes = [
        'S1 浴室跌倒 → 强推+短信+电话',
        'S2 蹲下捡物 → 3–5 秒延迟确认，不误报',
        'S3 卧室静止超时 → 推送提醒子女查看',
        'S4 连续 3 天活动量 < 基线 50% → 建议关注',
        'S5 首页卡片："昨夜睡眠 7.2h，起床 6:40，正常"',
        'S6 导出 30 天趋势 PDF 供就医参考',
    ]
    for i, sc in enumerate(scenes):
        put(s, rx+0.12, 2.32 + i*0.38, rw-0.24, 0.35, sc, 8, RGBColor(0x33,0x33,0x33), lh=1.15)

    # 硬边界
    rect(s, rx, 5.35, rw, 0.9, fill=INK)
    put(s, rx+0.12, 5.42, rw-0.24, 0.18, '硬边界', 8.5, WHITE, bold=True, spc=1.0)
    put(s, rx+0.12, 5.62, rw-0.24, 0.55,
        '无摄像头 / 无视频  ·  不做医学诊断  ·  不做 120 调度\n不测血压 / 血氧 / 血糖',
        7.5, RGBColor(0xCC,0xCC,0xCC), lh=1.2)

    footnote(s, '核心价值：无感 · 准 · 有预判 · 不骚扰。独居为核心设计场景，夫妻同居提供基础紧急报警。')

# ================= 第 6 页 · 技术架构（含架构流程图）=================
def p6(prs):
    s = blank(prs); bg(s, INK)
    header(s, 'The Core Tech · 统一隐空间世界模型', '06', dark=True)
    kicker(s, M, 0.58, 'THE CORE TECH', IKB_BRT)
    title(s, M, 0.78, 'ELPD 架构：Encoder → Latent Space → Predictor → Decoders', 18, WHITE)

    put(s, M, 1.22, 12.63, 0.35,
        '成品雷达模块在固件里算完，只输出"跌倒/未跌倒"，中间数据全部丢弃。家安不买"答案"，直接拿"原材料"——原始点云 WiFi 上云，跑自研深度模型。'
        'ELPD 用一个隐空间统一表征所有信号，所有任务从同一隐空间读取，信息零丢弃。',
        8.5, RGBColor(0xCC,0xCC,0xCC), lh=1.2)

    # ===== 架构流程图：卡片 + 箭头 =====
    # 四个核心卡片：Encoder → Latent Space → Predictor → Decoders
    card_y = 1.75; card_h = 1.55; card_w = 2.65; card_gap = 0.5
    total_w = card_w*4 + card_gap*3
    start_x = M + (12.63 - total_w)/2  # 居中

    arch_cards = [
        ('Encoder\n编码器', 'PointNet++ 变体\n时序 Transformer\n多雷达 Cross-Attention\n时间编码注入\n体征信号旁路\n\n→ 输出 S_t + Σ_t', IKB),
        ('Latent Space\n隐空间 S_t', '图结构：空间拓扑\n（节点/边/区域）\n向量：人状态\n全局上下文\n不确定性 Σ_t\n\n个人基线流形', RGBColor(0x00,0x40,0xC0)),
        ('Predictor\n预测器', 'Causal Transformer\nS_t → Ŝ_{t+1}\n多尺度预测\n(1s/10s/60s/3600s)\n\n自监督训练\n预测偏差 = 异常', IKB),
        ('Decoders\n解码器', 'Anomaly 异常检测\nNotification 通知\nReport 健康报告\nTracking 轨迹追踪\n\n轻量读出层\n不做重推理', IKB),
    ]
    card_positions = []
    for i, (ttl, desc, fill) in enumerate(arch_cards):
        x = start_x + i*(card_w + card_gap)
        card_positions.append((x, card_y, card_w, card_h))
        # Card background
        rect(s, x, card_y, card_w, card_h, fill=fill)
        # Title
        lines = ttl.split('\n')
        put(s, x+0.1, card_y+0.08, card_w-0.2, 0.35, lines[0], 10, WHITE, bold=True)
        if len(lines) > 1:
            put(s, x+0.1, card_y+0.38, card_w-0.2, 0.18, lines[1], 7.5, RGBColor(0xCC,0xD8,0xF0), spc=0.8)
        # Description
        put(s, x+0.1, card_y+0.58, card_w-0.2, card_h-0.65, desc, 6.8, RGBColor(0xDD,0xE4,0xF5), lh=1.15)

    # 箭头连接卡片
    arrow_y = card_y + card_h/2
    for i in range(3):
        x1 = card_positions[i][0] + card_w
        x2 = card_positions[i+1][0]
        mid_x = (x1 + x2) / 2
        # Horizontal line
        hline(s, x1, arrow_y, x2 - x1 - 0.08, IKB_BRT, 2)
        # Arrow head (small triangle via connector)
        ln = s.shapes.add_connector(1, Inches(x2-0.08), Inches(arrow_y), Inches(x2+0.02), Inches(arrow_y))
        ln.line.color.rgb = IKB_BRT; ln.line.width = Pt(2)
        ln.shadow.inherit = False
        # "→" label
        put(s, mid_x-0.12, arrow_y-0.55, 0.25, 0.18, '→', 12, WHITE, bold=True, align=PP_ALIGN.CENTER)

    # 输入标签（左侧）
    put(s, start_x - 2.6, card_y+0.15, 2.5, 0.45,
        '← 多模态输入\n  Radar Point Cloud\n  Time Encoding\n  (未来: Door, Temp)', 7, RGBColor(0xBB,0xBB,0xBB), lh=1.15, align=PP_ALIGN.RIGHT)
    # Input arrow
    hline(s, start_x-0.25, arrow_y, 0.28, IKB_BRT, 2)

    # 输出标签（右侧）
    put(s, start_x + total_w + 0.12, card_y+0.15, 2.5, 0.45,
        '→ 任务输出\n  Alert / Notification\n  Daily Report\n  Trajectory / Heatmap', 7, RGBColor(0xBB,0xBB,0xBB), lh=1.15)

    # ===== 自监督学习回环（下方弧形） =====
    loop_y = card_y + card_h + 0.12
    put(s, M, loop_y, 12.63, 0.2, '自监督学习回环：Predicted Ŝ_{t+1} vs Encoder(Real_PC_{t+1}) → gradient update  |  每个正常日子 = 训练数据，零标注', 7.5, IKB_BRT, bold=True, align=PP_ALIGN.CENTER, spc=0.8)
    # 回环示意线
    rect(s, start_x + card_w + card_gap, loop_y+0.22, card_w*2 + card_gap, 0.04, fill=IKB_BRT)

    # ===== 下方：传统管道 vs ELPD 对比 + 6信号 + 可行性 =====
    comp_y = loop_y + 0.45; comp_h = 1.05

    # 左：管道
    rect(s, M, comp_y, 6.15, comp_h, fill=INK_CARD)
    put(s, M+0.12, comp_y+0.06, 5.9, 0.16, '传统管道（竞品）', 8, GREY3, bold=True, spc=0.8)
    put(s, M+0.12, comp_y+0.25, 5.9, comp_h-0.3,
        '传感器→姿态分类→事件合并→状态机→基线统计→偏离检测→融合打分→通知路由  |  '
        '各模块输出离散标签，硬决策丢信息；"跌倒+浴室"=两个字符串，无空间上下文；规则冲突无法自修复；新功能=新模块+新数据',
        6.8, RGBColor(0x44,0x44,0x44), lh=1.18)

    # 右：ELPD 优势
    rect(s, M+6.35, comp_y, 6.28, comp_h, fill=IKB)
    put(s, M+6.47, comp_y+0.06, 6.0, 0.16, 'ELPD 核心差异', 8, WHITE, bold=True, spc=0.8)
    put(s, M+6.47, comp_y+0.25, 6.0, comp_h-0.3,
        '单一真相源：所有任务从同一隐空间读取，位置+姿态在同一向量空间  |  '
        '自监督零标注：每个正常日子都是训练信号  |  '
        '预测=异常：系统觉察"现实与预期不符"——规则引擎做不到  |  '
        '新任务=新Decoder：零额外标注，轻量读出层从现有隐空间读取',
        6.8, WHITE, lh=1.18)

    # 6 类信号 + 可行性
    hline(s, M, comp_y+comp_h+0.15, 12.63, RGBColor(0x55,0x55,0x55), 1)
    put(s, M, comp_y+comp_h+0.22, 12.63, 0.16, '从同一条点云流提取 6 类信号', 7.5, RGBColor(0xBB,0xBB,0xBB), spc=1.0)
    signals = [
        ('姿态序列·空间分布→跌倒初判/基线', '速度分布·径向速度→精细活动量化', '点云形变·PCA/质心→真摔vs坐下'),
        ('微多普勒谱·相位时频→步态参数(P2)', '生命体征·胸腔微动→呼吸/心率', '活动片段·时序聚类→基线时间分布'),
    ]
    for r, row in enumerate(signals):
        for c, sig in enumerate(row):
            put(s, M + c*4.2, comp_y+comp_h+0.42+r*0.3, 4.0, 0.25, sig, 7.2, RGBColor(0xAF,0xAF,0xAF))

    # 可行性
    hline(s, M, comp_y+comp_h+1.08, 12.63, RGBColor(0x55,0x55,0x55), 1)
    put(s, M, comp_y+comp_h+1.15, 12.63, 0.5,
        '技术可行性（行业验证/参考设计）：跌倒检测 92%+ 准确率、误报<3%（TI IWR6843 参考设计）| '
        '呼吸 ±0.4、心率 ±1.0 次/分（安静睡眠，西京学院 2026）| '
        '步态预测跌倒风险（Nature 2025/Frontiers 2024，学术阶段）\n'
        '硬件演进：IWR6843ISK 开发板 → TI 模组~200元 → 自研一体板~100元(量>5000) → 国产60GHz SoC~60元(量>5万，首选加特兰)。'
        '选型标准：点云质量+SDK开放度。诚实备注：自研Encoder精度是MVP最核心风险，并行跑固件作对照。',
        7, RGBColor(0x99,0x99,0x99), lh=1.18)

    footnote(s, '架构类比：就像 OpenAI 用 GPT 替代 N 个 NLP 模型——家安在居家养老场景用 ELPD 替代 N 个专用检测算法。', dark=True)

# ================= 第 7 页 · 算法清单 + 可行性 =================
def p7(prs):
    s = blank(prs); bg(s, PAPER)
    header(s, 'Algorithms & Feasibility · 算法与可行性', '07')
    kicker(s, M, 0.58, 'ALGORITHMS & FEASIBILITY')
    title(s, M, 0.78, '七个核心算法，站在已验证的能力上', 20)

    algs = [
        ('ALG-0', '点云→人体状态理解', '取代厂商固件，从原始点云直接输出连续状态向量而非离散标签'),
        ('ALG-1', '多雷达时序对齐+空间拓扑', '云端对齐多雷达时间线，自学习空间拓扑，零人工标注平面图'),
        ('ALG-2', '跌倒延迟确认', '3–5 秒观察窗口+多雷达交叉+个人基线，降误报核心'),
        ('ALG-3', '个人基线建模（14天）', '时序聚类提取典型日模式，稳健统计建立个人常态区间'),
        ('ALG-4', '异常偏离检测', '实时状态 vs 基线，多维度联合偏离评分'),
        ('ALG-5', '多信号贝叶斯置信度融合', '活动量↓+睡眠↓+心率↑ 同时出现时置信度指数叠加'),
        ('ALG-6', '分级通知决策', '异常分+持续性+紧急性→通知等级（不提/推送/强推/电话）'),
        ('ALG-7', '步态提取+跌倒风险预测', '微多普勒步态参数+ML，Phase 2 引入'),
    ]
    cw, ch = 3.05, 0.78
    gx, gy = 0.12, 0.1
    for i, (code, name, desc) in enumerate(algs):
        r, c = divmod(i, 4)
        x, y = M + c*(cw+gx), 1.35 + r*(ch+gy)
        accent = (i == 7)
        rect(s, x, y, cw, ch, fill=(IKB if accent else INK_CARD))
        tc = WHITE if accent else IKB
        td = WHITE if accent else RGBColor(0x33,0x33,0x33)
        put(s, x+0.1, y+0.06, cw-0.18, 0.18, f'{code}  {name}', 8, tc, bold=True, lh=1.1)
        put(s, x+0.1, y+0.38, cw-0.18, ch-0.44, desc, 7.2, td, lh=1.15)

    # 底部可行性
    hline(s, M, 2.98, 12.63, GREY2, 1)
    put(s, M, 3.1, 12.63, 0.18, '技术可行性（行业已验证/参考设计，非产品实测）', 8.5, IKB, bold=True, spc=1.0)
    feas = [
        ('跌倒检测 92%+ 准确率、误报<3%', 'TI IWR6843 参考设计，20㎡；Application Note ZHCAA43'),
        ('呼吸 ±0.4 次/分、心率 ±1.0 次/分', '安静睡眠状态；西京学院 2026，n=20'),
        ('步态参数预测跌倒风险', 'Nature Sci Rep 2025 / Frontiers in AI 2024，学术验证阶段'),
    ]
    for i, (metric, source) in enumerate(feas):
        x = M + i*4.2
        put(s, x, 3.32, 4.0, 0.25, metric, 9.5, INK, bold=True)
        put(s, x, 3.55, 4.0, 0.22, source, 7.5, GREY3, lh=1.1)

    # 硬件演进
    put(s, M, 3.9, 12.63, 0.18, '硬件演进路线（选型看点云质量+SDK开放度，不看固件精度）', 8.5, IKB, bold=True, spc=1.0)
    put(s, M, 4.1, 12.63, 0.65,
        'IWR6843ISK 开发板（$149，USB 直出点云，深度模型训练数据源）→ TI IWR6843 模组 ~200 元（MVP 产品）→ '
        '自研一体板 ~100 元（量 > 5,000，TI 同款芯片，单 PCB 集成 WiFi+电源）→ '
        '国产 60GHz SoC ~60 元（量 > 5 万，首选加特兰 Alps 系列。前提：SDK 开放原始点云输出，否则一票否决）\n'
        '各阶段点云特征分布一致，模型无缝迁移。国产替代候选：加特兰（出货量最大，支持级联 4D 成像）> 矽典微（超低功耗 AiP）> 岸达（成本最低）。',
        7.5, GREY3, lh=1.2)

    put(s, M, 5.0, 12.63, 0.18, 'MVP 技术风险与缓释', 8.5, IKB, bold=True, spc=1.0)
    put(s, M, 5.2, 12.63, 0.8,
        '① 合成数据→真实点云 domain gap 导致 Encoder 精度不达预期（高风险）：合成数据仅做基础预训练（"什么是人形"），真实数据自监督 fine-tune。'
        '开发期并行跑固件作对照基准，不达预期则先用固件过渡、自研并行。\n'
        '② 隐空间坍缩——大量"无人"帧导致模型只预测"人不在了"（高风险）：loss 加权非空帧/变化帧，curriculum learning 先学有人的片段。\n'
        '③ 自监督数据密度不足——MVP 阶段仅 2 雷达 5–8 户（中风险）：合成数据补充多样性；MVP 不要求 SOTA，比 naive baseline 好就是成功。',
        7.5, GREY3, lh=1.2)

    footnote(s, '上述精度为行业参考设计/学术验证，非家安产品实测。自研 Encoder 精度是 MVP 待验证核心指标。')

# ================= 第 8 页 · 护城河 =================
def p8(prs):
    s = blank(prs); bg(s, INK)
    header(s, 'The Moat · 为什么难复制', '08', dark=True)
    kicker(s, M, 0.58, 'WHY US', IKB_BRT)
    title(s, M, 0.78, '三层壁垒：范式 · 算法 · 时间', 20, WHITE)

    put(s, M, 1.3, 12.63, 0.4,
        '壁垒不在传感器（传感器是供应商的通用件），也不在单个算法（算法可以复现）。'
        '真正的壁垒在三样东西的叠加：架构范式的代际差异、跨学科团队的稀缺性、随时间复利加深的数据资产。',
        9, RGBColor(0xCC,0xCC,0xCC), lh=1.22)

    y = 1.85; h = 2.45; cw = 4.1; gx = 0.14
    layers = [
        ('LAYER 01 · 范式壁垒', '统一隐空间世界模型',
         '传统养老监测是管道架构——传感器→规则→事件→通知，每层做硬决策、丢信息。ELPD 用一个隐空间统一表征所有信号，多维度偏离在隐空间中自动耦合，不需外部融合。'
         '竞品从管道迁移到世界模型，要改的是公司架构，远不止代码。就像 OpenAI 用 GPT 替代 N 个 NLP 模型——家安在居家养老这个场景里做了同样的事。',
         '范式壁垒 · 架构代际差异', IKB),
        ('LAYER 02 · 算法壁垒', '自研点云 Encoder',
         '毫米波雷达点云极稀疏（每帧 10–50 点，LiDAR 数万点）、噪声特殊（多径鬼影、微多普勒模糊）、无颜色纹理。'
         '需要同时懂雷达信号处理 + 点云深度学习 + 自监督表示学习的团队。三个圈子的专家平时不坐在一起。'
         '竞品要么用厂商固件（硬件厂，回避了模型），要么搬视觉模型（AI 厂，回避了雷达信号）。',
         '算法壁垒 · 跨学科稀缺', GREY1),
        ('LAYER 03 · 时间壁垒', '数据飞轮 + 个人基线',
         '自监督学习零标注成本——每个正常日子都是训练信号。14 天建立个人基线，3 个月比 14 天准，1 年比 3 个月准。'
         '用得越久切换成本越高：个人基线是用户的数据资产，换品牌意味着重新学习 14 天。'
         '固件精度停在出厂那天，世界模型每天都在进化——这是时间的朋友，不是用钱能加速的。',
         '时间壁垒 · 越用越准', GREY1),
    ]
    for i, (no, t, d, tag, fill) in enumerate(layers):
        x = M + i*(cw+gx)
        dark = fill in (IKB, INK)
        rect(s, x, y, cw, h, fill=fill)
        tc = WHITE if dark else GREY3
        tt = WHITE if dark else INK
        td = RGBColor(0xEC,0xF0,0xFB) if fill == IKB else RGBColor(0x33,0x33,0x33)
        put(s, x+0.12, y+0.12, cw-0.2, 0.18, no, 7.5, tc, spc=1.0)
        put(s, x+0.12, y+0.35, cw-0.24, 0.28, t, 13, tt, bold=True)
        put(s, x+0.12, y+0.7, cw-0.24, 1.3, d, 7.5, td, lh=1.2)
        hline(s, x+0.12, y+h-0.35, cw-0.24, (RGBColor(0x88,0xA0,0xD8) if fill == IKB else GREY2), 1)
        put(s, x+0.12, y+h-0.28, cw-0.24, 0.18, tag, 7, tc, spc=0.8)

    put(s, M, 4.55, 12.63, 0.5,
        '数据飞轮：用户越多 → 数据越丰富 → 模型越强 → 产品越好 → 用户更多。每户每天产生 1–2GB 点云流，全部可作为自监督训练信号。'
        '飞轮一旦转起来，追赶需要的不只是钱，是时间。管道架构的竞品无论卖多少台，精度都停在出厂那天。',
        9, RGBColor(0xCC,0xCC,0xCC), lh=1.22)

    put(s, M, 5.2, 12.63, 0.4, '能力归属：传感器（TI/国产 SoC）= 供应商 commodity | 感知/聚合/分析/决策/交互 = 家安自研', 8, GREY3, spc=1.0)

    footnote(s, '三层壁垒的叠加效应：竞品单独突破任何一层都不够——需要同时具备架构能力+跨学科团队+时间积累，缺一不可。', dark=True)

# ================= 第 9 页 · 商业模式 =================
def p9(prs):
    s = blank(prs); bg(s, PAPER)
    header(s, 'Business Model · 商业模式', '09')
    kicker(s, M, 0.58, 'HOW WE MAKE MONEY')
    title(s, M, 0.78, '硬件获客 · 服务盈利', 20)

    # 左栏
    rect(s, M, 1.45, 5.8, 3.3, fill=INK_CARD)
    put(s, M+0.15, 1.55, 5.5, 0.2, '01 硬件层 · 买断获客', 9.5, GREY3, bold=True, spc=0.8)
    put(s, M+0.15, 1.85, 5.5, 0.4, '一次买断', 22, INK)
    put(s, M+0.15, 2.35, 5.5, 1.2,
        '299 / 499 / 799 元三档，用户永久拥有。\n'
        '加权 ASP 599 元 / BOM 425 元 / 硬件毛利 174 元（29%）。\n'
        '硬件微利——目的是把设备装进家，利润从服务来。',
        9, RGBColor(0x33,0x33,0x33), lh=1.25)
    hline(s, M+0.15, 3.7, 5.5, GREY2, 1)
    items_l = ['尝鲜版 299 元 — 1 雷达（仅浴室，尝鲜入口）',
               '全屋套 S 499 元 — 2 雷达（2室1厅，主推入门）',
               '全屋套 L 799 元 — 4 雷达（3室2厅，完整覆盖）',
               '硬件不指望赚钱，是获客入口',
               '低成本渠道把 CAC 压到 300 以内']
    for i, item in enumerate(items_l):
        put(s, M+0.15, 3.8 + i*0.3, 5.5, 0.26, f'— {item}', 8, RGBColor(0x33,0x33,0x33), lh=1.1)

    # 分隔线
    rect(s, 6.2, 1.45, 0.008, 3.3, fill=GREY2)

    # 右栏
    rect(s, 6.55, 1.45, 6.08, 3.3, fill=IKB)
    put(s, 6.7, 1.55, 5.78, 0.2, '02 服务层 · 订阅盈利', 9.5, WHITE, bold=True, spc=0.8)
    put(s, 6.7, 1.85, 5.78, 0.4, '199 元/年', 22, WHITE)
    put(s, 6.7, 2.35, 5.78, 1.2,
        '基础服务（实时告警、周报）永久免费。\n'
        '高级健康服务（月报 PDF、长期趋势、就医参考）按年订阅。\n'
        '毛利约 90%，免费层走轻量云、付费层承担重云成本。',
        9, WHITE, lh=1.25)
    hline(s, 6.7, 3.7, 5.78, RGBColor(0x88,0xA0,0xD8), 1)
    items_r = ['转化率保守取 15% — 子女为父母健康买单意愿高于一般 IoT（行业基准 5–10%）',
               '免费层走轻量云、付费层承担重云成本 — 订阅收入覆盖云成本有余',
               '越用越准 → 越不可替代 → 续费越稳']
    for i, item in enumerate(items_r):
        put(s, 6.7, 3.8 + i*0.3, 5.78, 0.26, f'— {item}', 8, WHITE, lh=1.1)

    # 底部单位经济
    hline(s, M, 5.05, 12.63, GREY2, 1)
    put(s, M, 5.15, 12.63, 0.2, '单位经济', 9.5, IKB, bold=True, spc=0.8)
    put(s, M, 5.4, 12.63, 0.8,
        '早期（BOM 425）：LTV 232 / CAC 300 = 0.77，承压。'
        '国产 SoC 降本后（BOM 110，硬件毛利 174→489）：LTV 547 / CAC 250 = 2.19，转正。'
        '盈亏平衡约在第 4 年。敏感性：订阅转化率 10%/15%/20% 对应 blended 订阅 LTV 约 39/58/77 元，降本仍是转正主要杠杆。\n'
        '第二曲线：沉淀的居家健康数据向保险/养老机构/医疗 B 端变现。已积累的个人基线数据和标注数据是 B 端服务的核心资产。',
        8.5, RGBColor(0x33,0x33,0x33), lh=1.22)

# ================= 第 10 页 · 财务预测 =================
def p10(prs):
    s = blank(prs); bg(s, PAPER)
    header(s, 'Financials · 财务预测', '10')
    kicker(s, M, 0.58, 'FINANCIALS · 中性务实')
    title(s, M, 0.78, '三年财务预测', 20)

    # KPI 行
    kpis = [
        ('Y3 年末装机', '1.5 万户', 'Y1 1,000 → Y2 5,000 → Y3 15,000', False),
        ('Y3 总收入', '637 万', '3 年累计约 953 万', False),
        ('毛利率（稳定）', '30%', '硬件+订阅加权平均', False),
        ('降本后 LTV/CAC', '2.19x', '国产 SoC 后转正', True),
    ]
    hline(s, M, 1.35, 12.63, GREY2, 1)
    for i, (lbl, num, desc, acc) in enumerate(kpis):
        x = M + i*3.16
        if i: rect(s, x-0.04, 1.42, 0.006, 1.1, fill=GREY2)
        nc = IKB if acc else INK
        put(s, x, 1.45, 3.0, 0.2, lbl, 7.5, GREY3, spc=0.8)
        put(s, x, 1.65, 3.0, 0.5, num, 20, nc)
        put(s, x, 2.18, 2.9, 0.25, desc, 7.5, GREY3, lh=1.1)

    # 收入柱状
    hline(s, M, 2.6, 12.63, GREY2, 1)
    put(s, M, 2.7, 12.63, 0.18, '三年收入', 8.5, INK, bold=True, spc=0.8)
    bars = [('Y1', 62.9, 0.099, False), ('Y2', 253.3, 0.398, False), ('Y3', 637.1, 1.0, True)]
    bw = 9.5; bx = 2.0; by = 2.92
    for i, (yr, val, frac, acc) in enumerate(bars):
        y = by + i*0.45
        put(s, M, y-0.02, 1.3, 0.22, f'{yr} 收入', 9, INK)
        rect(s, bx, y, bw, 0.2, fill=GREY1)
        rect(s, bx, y, bw*frac, 0.2, fill=(IKB if acc else RGBColor(0x33,0x33,0x33)))
        put(s, bx+bw+0.08, y-0.03, 1.5, 0.22, f'{val} 万', 10, INK)

    # 利润表
    put(s, M, 4.45, 12.63, 0.18, '三年利润简表（万元）| 净利 −206 / −355 / −532 万', 8.5, INK, bold=True, spc=0.8)
    # 简表
    rows = [
        ('', 'Y1', 'Y2', 'Y3'),
        ('硬件收入', '59.9', '239.6', '599.0'),
        ('订阅收入', '3.0', '13.7', '38.1'),
        ('总收入', '62.9', '253.3', '637.1'),
        ('总成本', '44.2', '178.3', '449.3'),
        ('毛利（率）', '18.7 (30%)', '75.0 (30%)', '187.8 (30%)'),
        ('总费用（人力+CAC+行政）', '225', '430', '720'),
        ('净利', '−206', '−355', '−532'),
    ]
    table_y = 4.7
    col_w = [2.8, 3.28, 3.28, 3.28]
    col_x = [M]
    for cw_i in col_w[:-1]:
        col_x.append(col_x[-1] + cw_i)
    for ri, row in enumerate(rows):
        y = table_y + ri*0.2
        is_header = (ri == 0)
        is_total = (ri == len(rows)-1)
        for ci, cell in enumerate(row):
            x = col_x[ci]; w = col_w[ci]
            c = INK if (is_header or is_total) else RGBColor(0x33,0x33,0x33)
            put(s, x, y, w, 0.18, cell, 7.5 if not is_header else 7.5, c, bold=(is_header or is_total), spc=0.6)

    # 说明
    put(s, M, 6.35, 12.63, 0.55,
        '前 3 年是投入期，亏损原因在规模不足——硬件毛利薄（174 元）而早期 CAC（300 元）高，靠融资补贴获客、占领 C 端空白、积累数据飞轮。'
        '盈亏平衡约在第 4 年，需要两件事先后发生：① 国产 SoC 降本（BOM 425→110），硬件毛利抬到 489 元；② 装机与订阅规模起来。'
        '天使轮资金正是覆盖到这个降本转正拐点——融的是过桥的钱。',
        8, GREY3, lh=1.2)

# ================= 第 11 页 · 竞争格局 =================
def p11(prs):
    s = blank(prs); bg(s, INK)
    header(s, 'Competition · 竞争格局', '11', dark=True)
    kicker(s, M, 0.58, 'THE WHITE SPACE', IKB_BRT)
    title(s, M, 0.78, 'C 端全屋买断，是一片空白', 20, WHITE)

    put(s, M, 1.3, 12.63, 0.5,
        '雷达跌倒检测已有厂商在做——点可、清雷、森思泰克、兆观。但都走 B 端项目制（养老院、医院）或卖模块给集成商。'
        '小米做智能家居（非康养），华为绑全屋装修，萤石核心是摄像头（雷达仅配件）。'
        '299–799 元、子女买给父母、15 分钟自装的 C 端全屋雷达套装——这个品类目前是空的。',
        9, RGBColor(0xCC,0xCC,0xCC), lh=1.22)

    y = 2.0; h = 2.6; cw = 4.1; gx = 0.14
    # 三列
    cols = [
        ('现有玩家', '固件 · 出厂即定型',
         '萤石/点可/森思泰克用雷达厂商固件，嵌入式 DSP 精度停在出厂那天。\n\n'
         '用户反馈："误报多、后来就不看了"——产品体验没到位，但独居老人跌倒检测的需求真实存在。\n\n'
         '固件方案多卖一万台不会变得更好。定价 500–2000+ 元，多需专业人员安装。',
         False),
        ('渠道错位', 'B 端项目 / 绑装修',
         '机构项目制针对养老院和医院，高端全屋方案走装修公司渠道。\n\n'
         '缺一个消费级的、子女回家自己就能装的东西——15 分钟、无需工具、价格在 500 元以内。\n\n'
         '价格带（299–799 vs 竞品 500–2000+）和安装门槛（自装 vs 上门）是核心差异点。',
         False),
        ('家安差异', 'ELPD 世界模型 · 越用越准',
         '云端深度模型持续进化，自监督学习让模型随数据积累变强——固件做不到这一点。\n\n'
         '14 天个人基线，用得越久越准。C 端买断零售 + 消费级安装体验。\n\n'
         '三点同时补空白：自研感知 + 全屋组网 + 订阅服务。',
         True),
    ]
    for i, (tag, ttl, desc, accent) in enumerate(cols):
        x = M + i*(cw+gx)
        rect(s, x, y, cw, h, fill=(IKB if accent else INK_CARD))
        nc = WHITE if accent else GREY3
        tc = WHITE if accent else INK
        dc = RGBColor(0xEC,0xF0,0xFB) if accent else RGBColor(0x33,0x33,0x33)
        put(s, x+0.12, y+0.12, cw-0.2, 0.18, tag, 8.5, nc, spc=1.0)
        put(s, x+0.12, y+0.35, cw-0.24, 0.3, ttl, 13, tc, bold=True)
        put(s, x+0.12, y+0.75, cw-0.24, h-0.9, desc, 7.5, dc, lh=1.22)

    # 底部海外对标
    put(s, M, 4.85, 12.63, 0.2, '海外对标', 9, WHITE, bold=True, spc=1.0)
    put(s, M, 5.08, 12.63, 0.8,
        'Vayyar Care（以色列）：$250/台 + $20/月订阅，估值约 $1B。CarePredict（美国）：$169/台 + $30/月。SafelyYou（美国）：纯 B 端，养老机构。'
        '三家模式均为"硬件+月费+B 端"。Apple Watch 跌倒检测纯 C 端买断，但 65+ 佩戴率低、夜间充电/洗澡取下——室内无感覆盖仍是雷达独占场景。'
        '国内暂无可比 C 端消费级雷达套装品牌。',
        8, RGBColor(0xBB,0xBB,0xBB), lh=1.22)

    footnote(s, '价格和形态是最强差异化武器。品牌壁垒形成后，新进入者需同时追上自研模型精度+数据飞轮规模+消费级渠道——三项同时从零开始。', dark=True)

# ================= 第 12 页 · 落地路径 =================
def p12(prs):
    s = blank(prs); bg(s, PAPER)
    header(s, 'Roadmap · 落地路径', '12')
    kicker(s, M, 0.58, 'ROADMAP · 落地与里程碑')
    title(s, M, 0.78, '落地路径与里程碑', 20)

    axis_y = 3.0; x0 = 1.8; span = 10.2
    hline(s, M, axis_y, 12.63, GREY2, 1)

    nodes = [
        ('M0–6 · MVP', '验证核心壁垒', '2 雷达起步（浴室+卧室）\n跌倒≥90%·误报<2次/周/户\n内测5–8户攒标注数据\n预算约100万', True, True),
        ('M6–12', '商用启动', '全屋套 S/L 上线\n1000 户·高级订阅上线\niOS 全功能', False, False),
        ('M12–18', '规模化+降本', '5000 户\n国产 SoC 降本样机\n步态预测·启动 A 轮', True, False),
        ('Phase 3 · 18M+', '单位经济转正', 'BOM 425→110\nLTV/CAC→2.19\nB 端数据合作', False, True),
        ('获客渠道', '低成本组合', '异地子女社群口碑\n适老化改造政府补贴\n养老机构/社区分销', True, False),
    ]
    for i, (yr, nm, ds, up, acc) in enumerate(nodes):
        cx = x0 + i*(span/4)
        sq = IKB if acc else INK
        rect(s, cx-0.06, axis_y-0.06, 0.12, 0.12, fill=sq)
        if up:
            put(s, cx-1.2, axis_y-1.25, 2.4, 0.2, yr, 8, (IKB if acc else GREY3), align=PP_ALIGN.CENTER, spc=0.6)
            put(s, cx-1.2, axis_y-1.0, 2.4, 0.25, nm, 10, (IKB if acc else INK), align=PP_ALIGN.CENTER)
            put(s, cx-1.2, axis_y-0.7, 2.4, 0.58, ds, 7.2, GREY3, align=PP_ALIGN.CENTER, lh=1.15)
        else:
            put(s, cx-1.2, axis_y+0.18, 2.4, 0.2, yr, 8, (IKB if acc else GREY3), align=PP_ALIGN.CENTER, spc=0.6)
            put(s, cx-1.2, axis_y+0.42, 2.4, 0.25, nm, 10, (IKB if acc else INK), align=PP_ALIGN.CENTER)
            put(s, cx-1.2, axis_y+0.72, 2.4, 0.58, ds, 7.2, GREY3, align=PP_ALIGN.CENTER, lh=1.15)

    # 预算明细
    hline(s, M, 4.05, 12.63, GREY2, 1)
    put(s, M, 4.15, 12.63, 0.18, 'MVP 阶段预算明细（M0–6，共约 100 万）', 8.5, IKB, bold=True, spc=0.8)
    budget_items = [
        '人力 90 万：产品+架构（创始人）· 嵌入式/边缘 1 人 · 后端 Python/时序库 1 人 · AI/ML 1 人 · iOS 1 人。5 人团队，月均约 15 万，6 个月。',
        '设备/云 10 万：IWR6843ISK 开发板 ×5、海凌科 LD6002 原型模组 ×10、云端 GPU 实例（训练+推理）、TimescaleDB 云实例。',
        'M1：传感器到货，数据 pipeline 跑通（雷达→云）。M2：L1 实时判别上线，App v0.1。M3：L2 基线学习训练完成，App 周报页。M4：内测 5–8 户，收集标注数据。M5：迭代降误报，扩展到 15–20 户。M6：效果评估、NPS 问卷、准备融资材料。',
    ]
    for i, item in enumerate(budget_items):
        put(s, M, 4.38 + i*0.3, 12.63, 0.26, item, 7.5, GREY3, lh=1.15)

    footnote(s, 'MVP 6 个月预算约 100 万，对应本轮融资第一阶段。关键验证指标：跌倒检测准确率≥90%、误报<2 次/周/户、NPS≥30。')

# ================= 第 13 页 · 团队 =================
def p13(prs):
    s = blank(prs); bg(s, PAPER)
    header(s, 'Team · 团队', '13')
    kicker(s, M, 0.58, 'WHY THIS TEAM')
    title(s, M, 0.78, '跨学科，是核心门槛', 20)

    put(s, M, 1.3, 12.63, 0.4,
        '这件事的技术门槛不在单个领域——在跨领域。雷达信号处理、点云深度学习、老年医学认知，三个圈子的专家平时不坐在一起。'
        '壁垒不在算法本身（算法可复现），在能同时驾驭这三个领域的团队组合 + 标注数据积累——这两样花钱短期堆不出来。',
        9, RGBColor(0x33,0x33,0x33), lh=1.22)

    y = 1.85; h = 2.4; cw = 4.1; gx = 0.14
    team_cards = [
        ('01 雷达信号处理',
         '从稀疏点云稳定提取多维特征，理解毫米波雷达物理模型、FMCW 信号链、噪声特性与多径鬼影剔除。'
         '关键能力：CFAR 检测、DoA 估计、微多普勒分析、点云聚类——将原始 ADC 数据转化为高质量点云流。',
         '核心技能：雷达信号链 · 嵌入式 · 点云预处理'),
        ('02 点云深度学习',
         '稀疏点云时序表示学习、端到端跌倒检测、微多普勒步态建模。云端 GPU 训练+推理。'
         '关键挑战：雷达点云和 RGB 图像是两种东西——极稀疏（10–50 点）、无纹理、噪声模型完全不同。不能把视觉模型直接搬过来。'
         '需自研适合雷达点云的 Encoder 架构（PointNet++ 变体 + 时序 Transformer）。',
         '核心技能：3D 深度学习 · 自监督学习 · 时序建模'),
        ('03 老年医学认知',
         '把呼吸/心率/活动/如厕/睡眠等居家数据翻译成"建议就医/关注"的临床关联，同时守住不做诊断的合规边界。'
         '关键能力：理解老年常见慢病（心衰/COPD/糖尿病/认知障碍）的居家可观测指标及其临床意义。'
         '产品设计确保所有输出有免责声明，不触发医疗器械监管。',
         '核心技能：老年医学 · 临床研究 · 医疗合规'),
    ]
    for i, (ttl, desc, skills) in enumerate(team_cards):
        x = M + i*(cw+gx)
        rect(s, x, y, cw, h, fill=INK_CARD)
        put(s, x+0.12, y+0.12, cw-0.2, 0.4, ttl, 18, IKB, bold=True)
        put(s, x+0.12, y+0.6, cw-0.24, 1.2, desc, 7.5, RGBColor(0x33,0x33,0x33), lh=1.2)
        hline(s, x+0.12, y+h-0.45, cw-0.24, GREY2, 1)
        put(s, x+0.12, y+h-0.38, cw-0.24, 0.28, skills, 7, GREY3, lh=1.1)

    hline(s, M, 4.5, 12.63, GREY2, 1)
    put(s, M, 4.6, 12.63, 0.2, 'MVP 团队配置 & 扩张计划', 8.5, IKB, bold=True, spc=0.8)
    put(s, M, 4.85, 12.63, 1.2,
        'MVP（5 人）：产品+架构（创始人）· 嵌入式/边缘 1 人 · 后端 Python/时序库 1 人 · AI/ML 1 人 · iOS 1 人。6 个月人力约 90 万。\n'
        'Y2（8 人）：+ 安卓 1 人 · + AI/ML 1 人 · + 运营/客户成功 1 人。年人力约 280 万。\n'
        'Y3（12 人）：+ 硬件工程师 1 人（自研一体板）· + 后端 1 人 · + 商务拓展 1 人 · + 医疗顾问 1 人（兼职）。年人力约 420 万。\n'
        '创始人背景：全栈工程师/产品架构，同时做产品和工程决策。团队采用远程优先，核心成员 base 国内一线城市。',
        7.5, GREY3, lh=1.2)

    footnote(s, '团队的核心竞争力不在人数——在跨学科组合的稀缺性。雷达+AI+医学三个方向各自找人容易，三个方向在同一个团队里协作很难。')

# ================= 第 14 页 · 融资计划 =================
def p14(prs):
    s = blank(prs); bg(s, INK)
    header(s, 'The Ask · 融资计划', '14', dark=True)
    kicker(s, M, 0.58, 'SEED ROUND', IKB_BRT)
    title(s, M, 0.78, '融资计划', 20, WHITE)

    y = 1.45; h = 2.5; cw = 3.05; gx = 0.12
    # 四卡
    cards = [
        ('融资额', '800 万', '人民币\n出让约 15–20%\n投前估值约 3200–4500 万', True),
        ('跑道', '18 个月', '覆盖 MVP 验证到\n国产 SoC 降本拐点\n约第 4 年盈亏平衡', False),
        ('资金用途', '', '研发/团队 50%（人力+GPU）\n供应链/流片 20%\n获客 20%\n云·数据·其他 10%', False),
        ('关键里程碑', '', '5,000 户装机\n国产 SoC 降本样机\n误报 <2 次/周/户\n启动 A 轮', False),
    ]
    for i, (lbl, num, desc, accent) in enumerate(cards):
        x = M + i*(cw+gx)
        rect(s, x, y, cw, h, fill=(IKB if accent else GREY1))
        nc = WHITE if accent else INK
        dc = RGBColor(0xEC,0xF0,0xFB) if accent else RGBColor(0x33,0x33,0x33)
        lc = WHITE if accent else GREY3
        put(s, x+0.12, y+0.15, cw-0.2, 0.2, lbl, 9, lc, spc=1.0)
        if num:
            put(s, x+0.12, y+0.45, cw-0.2, 0.5, num, 25, nc)
        put(s, x+0.12, y+1.1 if num else y+0.45, cw-0.2, h-1.3 if num else h-0.6, desc, 9, dc, lh=1.3)

    # 底部政策
    hline(s, M, 4.2, 12.63, RGBColor(0x55,0x55,0x55), 1)
    put(s, M, 4.3, 12.63, 0.22, '政策背书', 9, WHITE, bold=True, spc=1.0)
    put(s, M, 4.55, 12.63, 0.9,
        '国办发〔2024〕1 号明确"家庭配备智能安全监护设备"——产品品类即国家鼓励方向。\n'
        '毫米波雷达跌倒/生命体征监测已入工信部/民政部/卫健委《智慧健康养老产品及服务推广目录》（2022/2024），可争取申报入选。\n'
        '政策风向：2024 年"银发经济"首入政府工作报告，2025 年四部委联合推进智慧养老——监管框架正在形成，先入局者有合规先发优势。\n'
        '天使轮资金使用节奏：前 6 个月约 200 万（MVP 验证），6–12 个月约 300 万（商用启动+获客），12–18 个月约 300 万（规模化+降本）。',
        8, RGBColor(0xBB,0xBB,0xBB), lh=1.22)

    footnote(s, '本页数据基于中性务实假设。财务模型参数可调（见底稿输入区），投资人可自行调整假设并观察推导结果。', dark=True)

# ================= 第 15 页 · 愿景 =================
def p15(prs):
    s = blank(prs); bg(s, IKB)
    rect(s, 6.667, 0, 6.666, 7.5, fill=PAPER)

    # 左 IKB
    put(s, M, 0.18, 5, 0.22, 'Vision · 愿景', 8, WHITE, spc=1.2)
    kicker(s, M, 0.85, 'TEN-YEAR VISION', WHITE)

    tf = box(s, M, 1.25, 5.8, 1.8)
    for i, line in enumerate(['从跌倒监测切入，', '做居家养老的', '健康基础设施']):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = 1.08
        add_run(p, line, 24, WHITE, italic=(i == 2))

    put(s, M, 3.25, 5.8, 1.5,
        '中国进入深度老龄化，未来二十年数以亿计的老人在家养老。'
        '他们的健康对子女、对医疗养老体系长期是盲区。'
        '家安用持续、连续的居家健康数据把盲区补上——'
        '让每个在家变老的人被持续看见、被及时守护。'
        '这是我们要用十年做成的事。',
        9.5, RGBColor(0xDD,0xE4,0xF5), lh=1.3)

    put(s, M, 5.5, 5.8, 0.4, '家安 housafe · 天使轮 800 万', 9.5, WHITE, bold=True)

    # 右 PAPER
    rx = 7.2
    put(s, rx, 0.18, 5.5, 0.22, 'TAKEAWAYS', 8.5, GREY3, spc=1.2)
    takeaways = [
        ('01', '切入点足够痛',
         '2.2 亿老人、跌倒首位死因、现有方案全失效。独居老人背后的异地子女是一个被忽视的付费群体——他们有支付意愿，缺的是一个真正能用的产品。'),
        ('02', '壁垒足够深',
         'ELPD 世界模型 + 自研 Encoder + 数据飞轮。架构代际差异 + 跨学科团队稀缺性 + 时间复利效应——三重叠加，花钱短期追不上。'),
        ('03', '路径足够清',
         '买断获客（硬件微利建规模）→ 降本转正（国产 SoC 抬毛利）→ 数据变现（B 端健康数据服务）。三步自洽，拐点明确。'),
    ]
    for i, (n, t, d) in enumerate(takeaways):
        y = 1.2 + i*1.62
        put(s, rx, y, 5.5, 0.22, n, 9, IKB, bold=True, spc=1.0)
        put(s, rx, y+0.25, 5.5, 0.28, t, 14, INK)
        put(s, rx, y+0.55, 5.5, 0.65, d, 8.5, RGBColor(0x33,0x33,0x33), lh=1.22)

    put(s, rx, 6.85, 5.5, 0.22, '家安 housafe · 天使轮 800 万', 8, GREY3, align=PP_ALIGN.RIGHT, spc=0.8)

DECK = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15]

def new_prs():
    prs = Presentation()
    prs.slide_width = EMU_W; prs.slide_height = EMU_H
    return prs

def build_deck():
    prs = new_prs()
    for fn in DECK:
        fn(prs)
    prs.save('家安housafe_BP.pptx')
    print(f'成品: {len(DECK)} 页 → 家安housafe_BP.pptx')

if __name__ == '__main__':
    build_deck()
    print('OK')

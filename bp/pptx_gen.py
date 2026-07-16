# -*- coding: utf-8 -*-
"""家安 housafe BP → pptx 生成器
母版层（样式常量 + 绘制原语 + 母版模板.pptx）+ 内容层（15 页成品）。
视觉对齐 bp/阅读版.html（瑞士风 + 克莱因蓝 IKB）。
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

# ================= 母版层：视觉常量 =================
IKB       = RGBColor(0x00,0x2F,0xA7)
IKB_BRT   = RGBColor(0x5B,0x7B,0xFF)
INK       = RGBColor(0x0A,0x0A,0x0A)
PAPER     = RGBColor(0xFA,0xFA,0xF8)
GREY1     = RGBColor(0xF0,0xF0,0xEE)
GREY2     = RGBColor(0xD4,0xD4,0xD2)
GREY3     = RGBColor(0x73,0x73,0x73)
WHITE     = RGBColor(0xFF,0xFF,0xFF)
INK_CARD  = RGBColor(0xF5,0xF5,0xF4)   # 浅底卡片
W_60      = RGBColor(0xBF,0xC6,0xDB)   # 蓝底上的次要白（近似）

SANS = 'Arial'        # 西文/数字
ZH   = '微软雅黑'      # 中文

EMU_W, EMU_H = Inches(13.333), Inches(7.5)
M = 0.55              # 页边距

# ================= 母版层：绘制原语 =================
def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])

def bg(slide, color):
    r = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, EMU_W, EMU_H)
    r.fill.solid(); r.fill.fore_color.rgb = color
    r.line.fill.background(); r.shadow.inherit = False
    # 置底
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

def header(slide, left, right, dark=False):
    c = RGBColor(0xFF,0xFF,0xFF) if dark else GREY3
    op = 0.62 if dark else 1
    put(slide, M, 0.36, 8, 0.3, left, 9.5, c, latin=SANS, ea=ZH, spc=1.2)
    put(slide, 6.333, 0.36, 6.45, 0.3, right, 9.5, c, align=PP_ALIGN.RIGHT, latin=SANS, spc=1.2)

def kicker(slide, x, y, text, color=IKB):
    put(slide, x, y, 9, 0.3, text, 10.5, color, bold=True, latin=SANS, ea=ZH, spc=1.5)

def footnote(slide, text, dark=False, y=6.92):
    c = RGBColor(0xFF,0xFF,0xFF) if dark else GREY3
    put(slide, M, y, 12.23, 0.5, text, 9, c, latin=SANS, ea=ZH, lh=1.2)

def title(slide, x, y, text, size=30, color=INK):
    tf = box(slide, x, y, 12.2, size/40+0.6)
    p = tf.paragraphs[0]; p.line_spacing = 1.05
    for i, line in enumerate(text.split('\n')):
        if i: p = tf.add_paragraph(); p.line_spacing = 1.05
        add_run(p, line, size, color, bold=False)
    return tf

def card(slide, x, y, w, h, fill=INK_CARD):
    return rect(slide, x, y, w, h, fill=fill)

# ================= 内容层：15 页 =================
def p1(prs):  # 封面 IKB
    s = blank(prs); bg(s, IKB)
    header(s, '家安 housafe · 商业计划书', '2026 · 01 / 15', dark=True)
    kicker(s, M, 2.05, 'HOUSAFE · 居家养老 AI 健康守护', WHITE)
    tf = box(s, M, 2.55, 12, 2.0)
    for i, line in enumerate(['让每个在家变老的人，', '被持续看见']):
        p = tf.paragraphs[0] if i==0 else tf.add_paragraph()
        p.line_spacing = 1.08
        add_run(p, line, 40, WHITE, italic=(i==1))
    hline(s, M, 5.25, 12.23, RGBColor(0x6E,0x8A,0xC9), 1)
    put(s, M, 5.45, 11.5, 1.0,
        '面向异地子女的非接触式 AI 居家健康监测系统。毫米波雷达组网，无感采集老人在家的活动与生命体征，AI 识别异常、分级通知子女——从跌倒监测切入，做居家养老的健康基础设施。',
        12.5, WHITE, lh=1.35)
    put(s, M, 6.9, 7, 0.3, '天使轮 · 拟融资 800 万 / 18 个月跑道', 10, RGBColor(0xD5,0xDD,0xF2), spc=1.2)
    put(s, 8, 6.9, 4.78, 0.3, 'housafe · 2026', 10, RGBColor(0xD5,0xDD,0xF2), align=PP_ALIGN.RIGHT, spc=1.2)

def _sixcard(s, x, y, w, h, no, ttl, desc, accent=False):
    fill = IKB if accent else INK_CARD
    card(s, x, y, w, h, fill)
    tc = WHITE if accent else GREY3
    tb = WHITE if accent else INK
    td = RGBColor(0xEC,0xF0,0xFB) if accent else RGBColor(0x40,0x40,0x40)
    put(s, x+0.22, y+0.2, w-0.4, 0.3, no, 9.5, tc, bold=True, latin=SANS, spc=1.2)
    put(s, x+0.22, y+0.52, w-0.44, 0.5, ttl, 15, tb, bold=False)
    put(s, x+0.22, y+1.05, w-0.44, h-1.2, desc, 10, td, lh=1.3)

def p2(prs):  # 机会速览 六卡
    s = blank(prs); bg(s, PAPER)
    header(s, 'Executive Summary · 机会速览', '02 / 15')
    kicker(s, M, 0.85, 'THE OPPORTUNITY · 一页看懂')
    title(s, M, 1.15, '一页看懂家安 housafe', 27)
    cards = [
        ('01 痛点','2.2 亿老人的跌倒盲区','跌倒是 65+ 伤害死亡首位原因，约 30% 每年至少跌倒一次、年逾 4000 万人；47% 摔倒后无法自行起身，而子女异地。'),
        ('02 市场','万亿级银发经济','银发经济 2024 约 7 万亿 → 2035 约 30 万亿；智能养老设备 2025 约 1840 亿、年增约 20%；政策写进国办文件。'),
        ('03 产品','全屋雷达无感组网','2–4 个毫米波雷达组网，不穿戴、不操作、无摄像头；AI 识别跌倒/静止/体征异常/模式偏离四类事件。'),
        ('04 壁垒','自研点云 + 数据飞轮','不用厂商固件，直接从原始点云自研感知；多雷达聚合全屋画像；个人基线越用越准，形成时间锁。'),
        ('05 模式','买断获客 + 订阅盈利','硬件 299–799 元一次买断（微利获客），高级健康服务 199 元/年（高毛利）；单位经济随降本从 0.77 转正到 2.19。'),
        ('06 融资','天使轮 800 万','18 个月跑道，覆盖 MVP 验证到国产 SoC 降本样机，达成数千户商用与数据飞轮，为 A 轮铺路。'),
    ]
    cw, ch, gx, gy = 3.94, 2.18, 0.24, 0.22
    x0, y0 = M, 2.15
    for i,(no,t,d) in enumerate(cards):
        r,c = divmod(i,3)
        _sixcard(s, x0+c*(cw+gx), y0+r*(ch+gy), cw, ch, no, t, d, accent=(i==5))

def p3(prs):  # 痛点 深色
    s = blank(prs); bg(s, INK)
    header(s, 'The Problem · 需求痛点', '03 / 15', dark=True)
    kicker(s, M, 1.5, '被忽视的高频高危', WHITE)
    tf = box(s, M, 1.95, 5.5, 2.6)
    for i,line in enumerate(['每 3 位老人','就有 1 人','每年跌倒']):
        p = tf.paragraphs[0] if i==0 else tf.add_paragraph(); p.line_spacing=1.08
        add_run(p, line, 34, WHITE, italic=(i==2))
    put(s, M, 4.75, 5.4, 1.4,
        '我国 65+ 老人已达 2.2 亿（占 15.6%，2035 年约 3 亿），90% 居家养老。但父母在家过得好不好、有没有摔倒，对异地子女是一片盲区——空巢老人占比过半。',
        11, RGBColor(0xC8,0xC8,0xC8), lh=1.4)
    # 右列数据
    rx = 6.7
    hline(s, rx, 1.95, 6.08, RGBColor(0x55,0x55,0x55),1)
    put(s, rx, 2.05, 6, 0.3, '65+ 老人 · 年跌倒人数', 9.5, RGBColor(0xAAAAAA if False else 0x99,0x99,0x99), spc=1.2)
    put(s, rx, 2.28, 6, 0.9, '4000万+', 30, WHITE, latin=SANS)
    put(s, rx, 3.15, 6.08, 0.6, '约 30% 每年至少跌倒一次；跌倒是我国 65+ 老人伤害死亡首位原因、创伤性骨折首位原因。', 10.5, RGBColor(0xC8,0xC8,0xC8), lh=1.35)
    hline(s, rx, 3.95, 6.08, RGBColor(0x55,0x55,0x55),1)
    put(s, rx, 4.05, 6, 0.3, '摔倒后无法自行起身', 9.5, RGBColor(0x99,0x99,0x99), spc=1.2)
    put(s, rx, 4.28, 6, 0.9, '47%', 30, WHITE, latin=SANS)
    put(s, rx, 5.15, 6.08, 0.6, '20–30% 倒地超 1 小时；髋部骨折 1 年死亡率约 14–25%，人均住院约 1.98 万元。', 10.5, RGBColor(0xC8,0xC8,0xC8), lh=1.35)
    hline(s, rx, 5.9, 6.08, RGBColor(0x55,0x55,0x55),1)
    fx = [('摄像头：浴室不可装',rx),('手环：依从性30–50%',rx+3.05),('按钮：摔倒够不到',rx),('现有雷达：单房间/误报高',rx+3.05)]
    put(s, rx, 6.0, 3, 0.3, '摄像头：浴室不可装、隐私争议', 9.5, RGBColor(0xCF,0xCF,0xCF))
    put(s, rx+3.05, 6.0, 3, 0.3, '手环：依从性仅 30–50%', 9.5, RGBColor(0xCF,0xCF,0xCF))
    put(s, rx, 6.32, 3, 0.3, '按钮：摔倒够不到/昏迷无效', 9.5, RGBColor(0xCF,0xCF,0xCF))
    put(s, rx+3.05, 6.32, 3, 0.3, '现有雷达：单房间/误报高', 9.5, RGBColor(0xCF,0xCF,0xCF))
    footnote(s, '来源：民政部《2024 国家老龄事业发展公报》· 中国 CDC《老年人跌倒干预技术指南》(2011)/《核心信息》(2021) · BMJ 跌倒 Long-lie 研究', dark=True)

def _tower(s, x, base_y, w, h, lbl, num, sub, accent=False):
    y = base_y - h
    rect(s, x, y, w, h, fill=(IKB if accent else PAPER), line=(None if accent else GREY2), line_w=1)
    tc = WHITE if accent else GREY3
    nc = WHITE if accent else INK
    sc = RGBColor(0xDDE4F5 if False else 0xE0,0xE6,0xF5) if accent else GREY3
    put(s, x+0.18, y+0.18, w-0.3, 0.3, lbl, 9, tc, spc=1.1)
    put(s, x+0.18, y+0.5, w-0.3, 0.7, num, 25, nc, latin=SANS)
    put(s, x+0.18, y+h-0.62, w-0.32, 0.55, sub, 9, sc, lh=1.2)

def p4(prs):  # 市场 KPI塔
    s = blank(prs); bg(s, PAPER)
    header(s, 'The Market · 市场规模', '04 / 15')
    kicker(s, M, 0.8, 'WHY NOW · 万亿级赛道')
    title(s, M, 1.1, '一个正在爆发的银发健康市场', 25)
    base = 5.55; cw=2.9; gx=0.19; x0=M
    _tower(s, x0,          base, cw, 3.5, 'TAM · 银发经济 2035E', '30万亿', '2024 约 7 万亿 · CAGR≈15%', accent=True)
    _tower(s, x0+(cw+gx),  base, cw, 2.0, 'SAM · 智能养老设备', '≈2400亿', '设备 1840 亿 + 适老化 600 亿(2025)')
    _tower(s, x0+2*(cw+gx),base, cw, 2.7, '刚需人群 · 年跌倒', '4000万+', '65+ 老人跌倒人次 / 年')
    _tower(s, x0+3*(cw+gx),base, cw, 1.4, '需求信号 · 增速', '+53.6%', '健康监测可穿戴 25H1 出货')
    hline(s, M, 5.85, 12.23, GREY2, 1)
    forces = [('① 老龄化拐点','65+ 已 2.2 亿，深度老龄化'),('② 雷达降本','60GHz SoC 走向百元内'),
              ('③ AI 成熟','云端点云深度模型可行'),('④ 政策点名','国办发〔2024〕1 号写入')]
    for i,(a,b) in enumerate(forces):
        x = M + i*3.06
        put(s, x, 6.0, 3, 0.3, a, 10.5, INK, bold=True)
        put(s, x, 6.3, 3, 0.4, b, 9.5, GREY3, lh=1.2)
    footnote(s, '来源：《银发经济蓝皮书(2024)》社科文献出版社 · 中商产业研究院(2025) · IDC(2025H1)。市场金额为机构测算。')

def _step(s, x, y, w, no, t, d):
    hline(s, x, y, w-0.15, IKB, 2.2)
    put(s, x, y+0.12, w-0.2, 0.3, no, 10, GREY3, latin=SANS, spc=0.6)
    put(s, x, y+0.42, w-0.2, 0.5, t, 13, INK, bold=False)
    put(s, x, y+0.92, w-0.2, 1.8, d, 9.5, RGBColor(0x44,0x44,0x44), lh=1.3)

def p5(prs):  # 产品 流水线
    s = blank(prs); bg(s, PAPER)
    header(s, 'The Product · 产品', '05 / 15')
    kicker(s, M, 0.8, 'HOW IT WORKS')
    title(s, M, 1.1, '全屋雷达组网，无感守护', 25)
    put(s, M, 1.95, 12, 0.6, '子女带回家，一人操作、15 分钟装完、无需工具。雷达 7×24 采集，云端 AI 识别异常，异地子女第一时间知道；老人不穿戴、不操作、无感被守护。', 11, RGBColor(0x40,0x40,0x40), lh=1.35)
    steps = [('01 采集','毫米波雷达组网','Φ85mm 白色感应器，外观像烟雾探测器；天花板/墙壁磁吸，2.4G WiFi 直连云，功耗<2W；采集姿态、活动、呼吸、心率。'),
             ('02 感知','云端 AI 点云理解','原始点云上云，GPU 自研点云模型，识别跌倒、长时间静止、生命体征异常、日常模式偏离四类事件。'),
             ('03 判决','延迟确认降误报','跌倒初判后等 3–5 秒观察后续行为 + 多雷达交叉 + 个人基线，区分真摔与快速坐下/蹲下。'),
             ('04 通知','分级触达子女','不提 / 推送 / 强推+短信 / 电话四级；紧急事件（跌倒）同时短信与电话。')]
    cw=3.0; gx=0.11
    for i,(no,t,d) in enumerate(steps):
        _step(s, M+i*(cw+gx), 2.75, cw, no, t, d)
    hline(s, M, 5.75, 12.23, GREY2, 1)
    put(s, M, 5.9, 12.2, 0.6, '六大核心场景 ｜ S1 浴室跌倒强推 · S2 蹲下捡物不误报 · S3 卧室静止超时提醒 · S4 连续3天活动量<基线50%建议关注 · S5 首页"昨夜睡眠7.2h、起床6:40" · S6 导出30天趋势PDF供医生参考', 9.5, INK, lh=1.35)
    put(s, M, 6.62, 12.2, 0.4, '硬边界 ｜ 无摄像头/无视频 · 不做医学诊断 · 不做 120 调度 · 不测血压/血氧/血糖', 9.5, GREY3, lh=1.3)

def p6(prs):  # 技术A 深色
    s = blank(prs); bg(s, INK)
    header(s, 'The Core Tech · 自研点云感知', '06 / 15', dark=True)
    kicker(s, M, 0.8, 'THE CORE TECH', IKB_BRT)
    title(s, M, 1.1, '从点云到理解，全链路自研', 24, WHITE)
    put(s, M, 1.95, 12.2, 0.6, '成品雷达模块在固件里算完，只输出"跌倒/未跌倒"这类终点判断，把中间数据丢掉了。长期健康追踪要的恰恰是这些中间数据——我们不买"答案"，直接拿"原材料"。', 10.5, RGBColor(0xCC,0xCC,0xCC), lh=1.35)
    # 双卡
    card(s, M, 2.75, 6.0, 1.5, INK_CARD)
    put(s, M+0.22, 2.92, 5.6, 0.3, '成品固件', 9.5, GREY3, bold=True, spc=1.2)
    put(s, M+0.22, 3.25, 5.6, 0.9, '嵌入式 DSP · 输出终点判断 · 丢弃中间数据 · 受功耗约束 · 出厂即定型，精度停在出厂那天。', 10, INK, lh=1.35)
    card(s, M+6.23, 2.75, 6.0, 1.5, IKB)
    put(s, M+6.45, 2.92, 5.6, 0.3, '家安点云', 9.5, WHITE, bold=True, spc=1.2)
    put(s, M+6.45, 3.25, 5.6, 0.9, '原始点云上云 · 云端 GPU 深度时序模型 · 保留全维度信号 · 算法容量不受限 · 随标注数据持续进化。', 10, WHITE, lh=1.35)
    # 6 信号
    hline(s, M, 4.55, 12.23, RGBColor(0x55,0x55,0x55),1)
    put(s, M, 4.65, 10, 0.3, '从同一条点云流提取 6 类信号', 9.5, RGBColor(0xBB,0xBB,0xBB), spc=1.2)
    sig = [('姿态序列','← 空间分布 → 跌倒初判/基线'),('速度分布','← 径向速度 → 精细活动量化'),
           ('点云形变','← PCA/质心 → 真摔 vs 坐下'),('微多普勒谱','← 相位时频 → 步态参数(P2)'),
           ('生命体征','← 胸腔微动 → 呼吸/心率'),('活动片段','← 时序聚类 → 基线时间分布')]
    for i,(a,b) in enumerate(sig):
        r,c = divmod(i,3); x=M+c*4.08; y=5.05+r*0.62
        put(s, x, y, 4, 0.3, a, 10, WHITE, bold=True)
        put(s, x+1.15, y+0.02, 3, 0.3, b, 9, RGBColor(0xAF,0xAF,0xAF))
    footnote(s, '技术门槛：点云极稀疏(每帧 10–50 点)、多径鬼影、微多普勒模糊、无纹理——现有 3D 深度学习不能直接搬，需雷达信号处理 + 点云深度学习跨学科团队。', dark=True)

def p7(prs):  # 技术B 浅色 ALG
    s = blank(prs); bg(s, PAPER)
    header(s, 'Algorithms & Feasibility · 算法与可行性', '07 / 15')
    kicker(s, M, 0.8, 'ALGORITHMS & FEASIBILITY')
    title(s, M, 1.1, '七个算法，站在已验证的能力上', 24)
    algs = [('ALG-0','点云→人体状态理解'),('ALG-1','多雷达对齐+空间拓扑聚合'),('ALG-2','跌倒延迟确认（降误报）'),('ALG-3','个人基线建模（14天）'),
            ('ALG-4','异常偏离检测'),('ALG-5','多信号贝叶斯融合'),('ALG-6','分级通知决策'),('ALG-7','步态提取+跌倒预测(P2)')]
    cw=2.9; gx=0.19; ch=0.86
    for i,(a,b) in enumerate(algs):
        r,c=divmod(i,4); x=M+c*(cw+gx); y=1.95+r*(ch+0.18)
        acc = (i==7)
        card(s, x, y, cw, ch, IKB if acc else INK_CARD)
        put(s, x+0.16, y+0.12, cw-0.3, 0.3, a, 9.5, WHITE if acc else IKB, bold=True, latin=SANS, spc=0.8)
        put(s, x+0.16, y+0.42, cw-0.3, 0.4, b, 10, WHITE if acc else INK, lh=1.15)
    hline(s, M, 4.05, 12.23, GREY2, 1)
    feas = [('跌倒检测','92%+','误报<3%（TI IWR6843 参考设计，20㎡）'),
            ('呼吸/心率误差','±0.4','心率 ±1.0 次/分（安静睡眠，西京学院2026）'),
            ('步态预测跌倒风险','已验证','Nature Sci Rep 2025 / Frontiers 2024，学术阶段')]
    for i,(l,n,d) in enumerate(feas):
        x=M+i*4.08
        put(s, x, 4.25, 3.9, 0.3, l, 9.5, GREY3, spc=1.1)
        put(s, x, 4.55, 3.9, 0.7, n, 22, INK, latin=SANS)
        put(s, x, 5.35, 3.9, 0.6, d, 9, GREY3, lh=1.25)
    footnote(s, '硬件演进（看点云质量+SDK开放度）：IWR6843 开发板 → TI 模组 ~200 元 → 自研一体板 ~100 元 → 国产 60GHz SoC ~60 元（加特兰）。上述精度为行业参考设计/学术验证，非产品实测；自研管线精度是 MVP 待验证核心。', y=6.75)

def _layer(s, x, y, w, h, no, t, d, tag, fill):
    dark = fill in (IKB, INK)
    rect(s, x, y, w, h, fill=fill)
    tc = WHITE if dark else GREY3
    tt = WHITE if dark else INK
    td = RGBColor(0xEC,0xF0,0xFB) if fill==IKB else (RGBColor(0x44,0x44,0x44))
    put(s, x+0.22, y+0.22, w-0.4, 0.3, no, 9, tc, spc=1.1)
    put(s, x+0.22, y+0.62, w-0.44, 0.5, t, 17, tt, bold=False)
    put(s, x+0.22, y+1.25, w-0.44, 2.0, d, 9.5, td, lh=1.35)
    hline(s, x+0.22, y+h-0.5, w-0.5, (RGBColor(0x88,0xA0,0xD8) if fill==IKB else GREY2), 1)
    put(s, x+0.22, y+h-0.4, w-0.44, 0.3, tag, 8.5, tc, spc=1.0)

def p8(prs):  # 护城河 深色 三层
    s = blank(prs); bg(s, INK)
    header(s, 'The Moat · 为什么难复制', '08 / 15', dark=True)
    kicker(s, M, 0.8, 'WHY US', IKB_BRT)
    title(s, M, 1.1, '壁垒：自研的三层软件', 25, WHITE)
    y=2.15; h=3.5; cw=3.94; gx=0.24
    _layer(s, M,            y, cw, h, 'LAYER 01 · 感知层', '自研点云→跌倒', '真正的算法壁垒。门槛不在姿态分类，而在积累跨场景、跨时长的点云标注数据 + 跨学科团队——这两样花钱短期难复制。', '算法壁垒', IKB)
    _layer(s, M+(cw+gx),    y, cw, h, 'LAYER 02 · 聚合层', '多雷达数据聚合', '产品/架构壁垒。把 N 条独立时间线拼成同一老人的全屋行为序列、空间拓扑自学习、跨房间追踪；独立模块架构不支持跨设备关联。', '架构壁垒', GREY1)
    _layer(s, M+2*(cw+gx),  y, cw, h, 'LAYER 03 · 分析层', '个人基线 + 融合', '时间壁垒。14 天学出个人常态，以偏离自身基线判断；新进入者上线前 14 天无基线可用，用得越久切换成本越高。', '时间壁垒 · 越用越准', GREY1)
    put(s, M, 6.0, 12.2, 0.7, '数据飞轮：用户越多、用得越久 → 标注数据与个人基线越丰富 → 模型越准 → 体验越好 → 用户越多。壁垒随使用复利加深，而固件方案停在出厂那天。', 10, RGBColor(0xC8,0xC8,0xC8), lh=1.4)

def _duo(s, x, y, w, tag, ttl, desc, items, accent=False):
    col = IKB if accent else GREY3
    put(s, x, y, w, 0.3, tag, 10, col, bold=True, latin=SANS, spc=0.8)
    put(s, x, y+0.35, w, 0.7, ttl, 27, IKB if accent else INK, latin=ZH)
    put(s, x, y+1.35, w, 0.9, desc, 11, RGBColor(0x40,0x40,0x40), lh=1.35)
    hline(s, x, y+2.5, w, GREY2, 1)
    for i,it in enumerate(items):
        put(s, x, y+2.62+i*0.36, w, 0.35, '— '+it, 9.5, RGBColor(0x33,0x33,0x33), lh=1.2)

def p9(prs):  # 商业模式 浅色 双栏
    s = blank(prs); bg(s, PAPER)
    header(s, 'Business Model · 商业模式', '09 / 15')
    kicker(s, M, 0.8, 'HOW WE MAKE MONEY')
    title(s, M, 1.1, '硬件获客 · 服务盈利', 25)
    _duo(s, M, 2.15, 5.6, '01 硬件层 · 买断获客', '一次买断',
         '299 / 499 / 799 元三档，用户永久拥有；加权 ASP 599 / BOM 425 / 毛利 174 元（29%）。定位微利把设备装进家。',
         ['尝鲜 299 · 全屋 S 499 · 全屋 L 799','硬件不指望赚钱，是获客入口','低成本渠道把 CAC 压到 300 以内'])
    rect(s, 6.62, 2.15, 0.012, 3.4, fill=GREY2)
    _duo(s, 7.05, 2.15, 5.6, '02 服务层 · 订阅盈利', '199 元/年',
         '基础服务（实时告警、周报）永久免费；高级服务（月报PDF、长期趋势、就医参考）订阅，毛利约 90%。',
         ['免费层轻云、付费层承担重云成本','转化率保守取 15%（子女买单）','越用越准 → 越不可替代'], accent=True)
    hline(s, M, 5.95, 12.23, GREY2, 1)
    tf=box(s, M, 6.1, 12.2, 0.8); p=tf.paragraphs[0]; p.line_spacing=1.4
    add_run(p,'单位经济：',11,INK,bold=True); add_run(p,'早期(BOM425) LTV 232 / CAC 300 = 0.77 承压 → 国产SoC降本后(BOM110) LTV 547 / CAC 250 = ',11,RGBColor(0x40,0x40,0x40))
    add_run(p,'2.19 转正',11,IKB,bold=True); add_run(p,'。第二曲线：数据向保险/养老机构/医疗 B 端变现。',11,RGBColor(0x40,0x40,0x40))

def p10(prs):  # 财务 浅色
    s = blank(prs); bg(s, PAPER)
    header(s, 'Financials · 财务预测', '10 / 15')
    kicker(s, M, 0.8, 'FINANCIALS · 中性务实')
    title(s, M, 1.1, '三年财务预测', 24)
    kpis=[('Y3 年末装机','1.5万户','Y1 1k→Y2 5k→Y3 15k'),('Y3 总收入','637万','3 年累计约 953 万'),
          ('毛利率（稳定）','30%','硬件 + 订阅加权'),('降本后单位经济','2.19x','LTV/CAC · 国产 SoC 后')]
    hline(s, M, 2.0, 12.23, GREY2, 1)
    for i,(l,n,d) in enumerate(kpis):
        x=M+i*3.06
        if i: rect(s, x-0.05, 2.1, 0.008, 1.3, fill=GREY2)
        put(s, x, 2.15, 3, 0.3, l, 9, GREY3, spc=1.0)
        put(s, x, 2.45, 3, 0.7, n, 22, INK, latin=SANS)
        put(s, x, 3.25, 2.9, 0.4, d, 9, GREY3, lh=1.2)
    # 柱状
    bars=[('Y1 收入',0.099,'62.9万',False),('Y2 收入',0.397,'253万',False),('Y3 收入',1.0,'637万',True)]
    by=3.95; bw=8.2; bx=M+1.5
    for i,(l,frac,val,acc) in enumerate(bars):
        y=by+i*0.62
        put(s, M, y-0.02, 1.4, 0.3, l, 10, INK)
        rect(s, bx, y, bw, 0.26, fill=GREY1)
        rect(s, bx, y, bw*frac, 0.26, fill=(IKB if acc else RGBColor(0x33,0x33,0x33)))
        put(s, bx+bw+0.1, y-0.05, 1.4, 0.3, val, 12, INK, latin=SANS)
    hline(s, M, 5.95, 12.23, GREY2, 1)
    put(s, M, 6.1, 12.2, 0.9, '前 3 年是投入期，净利 −206 / −355 / −532 万——不是单位经济问题，是规模不足：硬件毛利薄(174元)而早期 CAC(300元)高，靠融资补贴获客、占领 C 端空白、积累数据。盈亏平衡出现在①国产 SoC 降本(BOM425→110，硬件毛利抬到489元)②装机与订阅规模起来之后(约第4年)。天使轮资金正是覆盖到降本转正拐点。', 9.5, RGBColor(0x40,0x40,0x40), lh=1.35)

def _c3(s, x, y, w, h, tag, ttl, desc, accent=False):
    card(s, x, y, w, h, IKB if accent else INK_CARD)
    put(s, x+0.22, y+0.22, w-0.4, 0.3, tag, 9.5, WHITE if accent else GREY3, spc=1.1)
    put(s, x+0.22, y+0.6, w-0.44, 0.5, ttl, 16, WHITE if accent else INK)
    put(s, x+0.22, y+1.15, w-0.44, h-1.3, desc, 9.5, (RGBColor(0xEC,0xF0,0xFB) if accent else RGBColor(0x44,0x44,0x44)), lh=1.35)

def p11(prs):  # 竞争 深色
    s = blank(prs); bg(s, INK)
    header(s, 'Competition · 竞争格局', '11 / 15', dark=True)
    kicker(s, M, 0.8, 'THE WHITE SPACE', IKB_BRT)
    title(s, M, 1.1, 'C 端全屋买断，是一片空白', 24, WHITE)
    put(s, M, 1.95, 12.2, 0.6, '价格与形态是最强差异点：子女买给父母、15 分钟自装的 C 端全屋雷达套装几乎没有对手；雷达厂商走 B 端项目制，小米做智能家居非康养，华为绑装修，萤石以摄像头为主。', 10.5, RGBColor(0xCC,0xCC,0xCC), lh=1.35)
    y=2.75; h=2.7; cw=3.94; gx=0.24
    _c3(s, M,           y, cw, h, '现有玩家', '固件 · 出厂即定型', '萤石/点可/森思泰克用雷达厂商固件，嵌入式 DSP 精度停在出厂那天，用户反馈"误报多、后来就不看了"。')
    _c3(s, M+(cw+gx),   y, cw, h, '渠道错位', 'B 端项目 / 绑装修', '多为机构项目制或高端全屋，缺一个消费级、子女能自己装的产品。')
    _c3(s, M+2*(cw+gx), y, cw, h, '家安差异', '自研点云 · 越用越准', '云端深度模型持续进化 + 个人基线 + C 端买断零售，三点同时补缺口。', accent=True)
    footnote(s, '海外对标均为"硬件+月费+B 端"（Vayyar Care $250+$20/月、估值约 $1B；CarePredict $169+$30/月；SafelyYou 纯 B 端）；Apple Watch 纯 C 端但 65+ 佩戴率低、夜间/洗澡失效——室内无感覆盖仍是雷达独占。', dark=True)

def p12(prs):  # 落地 时间线
    s = blank(prs); bg(s, PAPER)
    header(s, 'Roadmap · 落地路径', '12 / 15')
    kicker(s, M, 0.8, 'ROADMAP · 落地与里程碑')
    title(s, M, 1.1, '落地路径与里程碑', 24)
    axis_y=4.0; x0=1.7; span=9.8
    hline(s, M, axis_y, 12.23, GREY2, 1)
    nodes=[('M0–6 · MVP','验证核心壁垒','2雷达·跌倒≥90%·误报<2次/周·内测5–8户',True,True),
           ('M6–12','商用启动','全屋套上线·1000户·高级订阅·iOS',False,False),
           ('M12–18','规模化+降本','5000户·国产SoC样机·步态预测·启动A轮',True,False),
           ('Phase 3 · 18M+','单位经济转正','BOM425→110·LTV/CAC→2.19·B端数据',False,True),
           ('获客渠道','低成本组合','子女社群·适老化补贴·机构分销',True,False)]
    for i,(yr,nm,ds,up,acc) in enumerate(nodes):
        cx = x0 + i*(span/4)
        sq = IKB if acc else INK
        rect(s, cx-0.06, axis_y-0.06, 0.12, 0.12, fill=sq)
        if up:
            put(s, cx-1.15, axis_y-1.35, 2.3, 0.3, yr, 9, (IKB if acc else GREY3), align=PP_ALIGN.CENTER, latin=SANS, spc=0.6)
            put(s, cx-1.15, axis_y-1.05, 2.3, 0.35, nm, 12, (IKB if acc else INK), align=PP_ALIGN.CENTER)
            put(s, cx-1.2, axis_y-0.62, 2.4, 0.5, ds, 8.5, GREY3, align=PP_ALIGN.CENTER, lh=1.2)
        else:
            put(s, cx-1.15, axis_y+0.22, 2.3, 0.3, yr, 9, (IKB if acc else GREY3), align=PP_ALIGN.CENTER, latin=SANS, spc=0.6)
            put(s, cx-1.15, axis_y+0.52, 2.3, 0.35, nm, 12, (IKB if acc else INK), align=PP_ALIGN.CENTER)
            put(s, cx-1.2, axis_y+0.95, 2.4, 0.5, ds, 8.5, GREY3, align=PP_ALIGN.CENTER, lh=1.2)
    footnote(s, 'MVP 6 个月预算约 100 万（人力 90 万 + 设备/云 10 万），对应本轮融资第一阶段。')

def p13(prs):  # 团队 浅色
    s = blank(prs); bg(s, PAPER)
    header(s, 'Team · 团队', '13 / 15')
    kicker(s, M, 0.8, 'WHY THIS TEAM')
    title(s, M, 1.1, '跨学科，是核心门槛', 25)
    put(s, M, 1.95, 12.2, 0.5, '壁垒的核心不在算法本身（算法可复现），而在标注数据积累 + 三种能力的团队组合——花钱短期难复制。', 10.5, RGBColor(0x40,0x40,0x40), lh=1.3)
    tri=[('01','雷达信号处理','从稀疏点云稳定提取多维特征，理解毫米波雷达物理、噪声模型与多径鬼影。'),
         ('02','点云深度学习','稀疏点云时序表示学习、端到端跌倒检测、微多普勒步态建模，云端 GPU 深度模型。'),
         ('03','老年医学认知','把呼吸/心率/活动/如厕指标翻译成"建议就医/关注"的临床关联，守住不做诊断的合规边界。')]
    y=2.7; h=2.6; cw=3.94; gx=0.24
    for i,(no,t,d) in enumerate(tri):
        x=M+i*(cw+gx)
        card(s, x, y, cw, h, INK_CARD)
        put(s, x+0.22, y+0.24, cw-0.4, 0.8, no, 30, IKB, latin=SANS)
        put(s, x+0.22, y+1.15, cw-0.44, 0.5, t, 15, INK)
        put(s, x+0.22, y+1.65, cw-0.44, 0.9, d, 9.5, RGBColor(0x44,0x44,0x44), lh=1.35)
    hline(s, M, 5.6, 12.23, GREY2, 1)
    put(s, M, 5.75, 12.2, 0.6, 'MVP 团队 5 人 ｜ 产品+架构（创始人）· 嵌入式/边缘 · 后端（Python/时序库）· AI/ML · iOS；6 个月人力约 90 万，第 2/3 年扩至 8/12 人。', 10, INK, lh=1.35)

def p14(prs):  # 融资 深色
    s = blank(prs); bg(s, INK)
    header(s, 'The Ask · 融资计划', '14 / 15', dark=True)
    kicker(s, M, 0.8, 'SEED ROUND', IKB_BRT)
    title(s, M, 1.1, '融资计划', 25, WHITE)
    y=2.15; h=3.0; cw=2.9; gx=0.19
    # 卡1 IKB
    rect(s, M, y, cw, h, fill=IKB)
    put(s, M+0.2, y+0.3, cw-0.4, 0.3, '融资额', 9, WHITE, spc=1.1)
    put(s, M+0.2, y+0.9, cw-0.4, 0.9, '800万', 30, WHITE, latin=SANS)
    put(s, M+0.2, y+1.9, cw-0.4, 0.9, '人民币 · 出让约 15–20%（投前估值约 3200–4500 万）', 9.5, RGBColor(0xEC,0xF0,0xFB), lh=1.3)
    # 卡2 灰
    rect(s, M+(cw+gx), y, cw, h, fill=GREY1)
    put(s, M+(cw+gx)+0.2, y+0.3, cw-0.4, 0.3, '跑道', 9, GREY3, spc=1.1)
    put(s, M+(cw+gx)+0.2, y+0.9, cw-0.4, 0.9, '18个月', 28, INK, latin=SANS)
    put(s, M+(cw+gx)+0.2, y+1.9, cw-0.4, 0.9, '覆盖 MVP 验证到国产 SoC 降本拐点', 9.5, RGBColor(0x44,0x44,0x44), lh=1.3)
    # 卡3 用途
    rect(s, M+2*(cw+gx), y, cw, h, fill=GREY1)
    put(s, M+2*(cw+gx)+0.2, y+0.3, cw-0.4, 0.3, '资金用途', 9, GREY3, spc=1.1)
    put(s, M+2*(cw+gx)+0.2, y+0.8, cw-0.4, 2.0, '研发/团队 50%\n供应链 / 流片 20%\n获客 20%\n云 · 数据 · 其他 10%', 11, INK, lh=1.5)
    # 卡4 里程碑
    rect(s, M+3*(cw+gx), y, cw, h, fill=GREY1)
    put(s, M+3*(cw+gx)+0.2, y+0.3, cw-0.4, 0.3, '关键里程碑', 9, GREY3, spc=1.1)
    put(s, M+3*(cw+gx)+0.2, y+0.8, cw-0.4, 2.0, '5000 户装机\n国产 SoC 降本样机\n误报<2次/周/户\n启动 A 轮', 11, INK, lh=1.5)
    footnote(s, '政策背书：国办发〔2024〕1 号明确"家庭配备智能安全监护设备"；毫米波雷达跌倒/生命体征监测已入工信部/民政部/卫健委《智慧健康养老产品及服务推广目录》——产品即国家鼓励品类，可争取申报入选。', dark=True)

def p15(prs):  # 愿景 IKB分屏
    s = blank(prs); bg(s, IKB)
    rect(s, 6.667, 0, 6.666, 7.5, fill=PAPER)
    # 左 IKB
    put(s, M, 0.36, 5, 0.3, 'Vision · 愿景', 9.5, RGBColor(0xFF,0xFF,0xFF), spc=1.2)
    kicker(s, M, 1.5, 'TEN-YEAR VISION', WHITE)
    tf=box(s, M, 1.95, 5.4, 2.2)
    for i,line in enumerate(['从跌倒监测切入，','做居家养老的','健康基础设施']):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph(); p.line_spacing=1.1
        add_run(p, line, 27, WHITE, italic=(i==2))
    put(s, M, 4.4, 5.4, 1.5, '中国正进入深度老龄化，未来二十年数以亿计的老人在家养老，他们的健康对子女、对医疗养老体系长期是盲区。家安用持续、连续的居家健康数据把盲区补上。', 10, RGBColor(0xDD,0xE4,0xF5), lh=1.4)
    put(s, M, 6.5, 5.4, 0.5, '让每个在家变老的人，被持续看见、被及时守护。', 11, WHITE)
    # 右 白
    rx=7.4
    put(s, rx, 0.36, 5, 0.3, 'TAKEAWAYS', 9.5, GREY3, spc=1.2)
    tk=[('01','切入点足够痛','2.2 亿老人、跌倒首位死因、现有方案全失效。'),
        ('02','壁垒足够深','自研点云 + 数据飞轮，越用越准、短期难复制。'),
        ('03','路径足够清','买断获客 → 降本转正 → 数据变现，自洽可信。')]
    for i,(n,t,d) in enumerate(tk):
        y=1.6+i*1.5
        put(s, rx, y, 5, 0.3, n, 10, IKB, bold=True, latin=SANS, spc=1.0)
        put(s, rx, y+0.32, 5, 0.4, t, 15, INK)
        put(s, rx, y+0.78, 5, 0.5, d, 9.5, RGBColor(0x44,0x44,0x44), lh=1.3)
    put(s, rx, 6.9, 5.3, 0.3, '家安 housafe · 天使轮 800 万', 9, GREY3, align=PP_ALIGN.RIGHT, spc=1.0)

DECK = [p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15]

def new_prs():
    prs = Presentation()
    prs.slide_width = EMU_W; prs.slide_height = EMU_H
    return prs

def build_deck():
    prs = new_prs()
    for fn in DECK: fn(prs)
    prs.save('家安housafe_BP.pptx')
    print('成品:', len(prs.slides.__iter__.__self__._sldIdLst) if False else len(DECK), '页 → 家安housafe_BP.pptx')

# ========== 母版模板文件（示范版式 + 样式规范） ==========
def build_master():
    prs = new_prs()
    # 1 封面版式
    p1(prs)
    # 2 浅色内容版式（示范）
    s = blank(prs); bg(s, PAPER); header(s,'浅色内容版式 · LIGHT LAYOUT','LAYOUT / LIGHT')
    kicker(s, M, 0.8, 'KICKER · 章节英文标签'); title(s, M, 1.1, '主标题占位 · 无衬线细体', 26)
    for i in range(3):
        x=M+i*4.18; card(s,x,2.4,3.9,2.4,INK_CARD)
        put(s,x+0.22,2.62,3.5,0.3,f'0{i+1} 卡片标签',9.5,GREY3,spc=1.1)
        put(s,x+0.22,3.0,3.5,0.4,'卡片标题占位',15,INK)
        put(s,x+0.22,3.55,3.5,1.0,'正文占位。瑞士风：直角纯色块 + 无衬线 + 单一 IKB 强调色 + 发丝线。',10,RGBColor(0x44,0x44,0x44),lh=1.35)
    footnote(s,'来源脚注版式 · 数据来源统一放这里，9pt 灰字')
    # 3 深色内容版式
    s = blank(prs); bg(s, INK); header(s,'深色内容版式 · DARK LAYOUT','LAYOUT / DARK',dark=True)
    kicker(s, M, 0.8, 'KICKER', IKB_BRT); title(s, M, 1.1, '深色页用于痛点/技术/护城河/竞争/融资', 24, WHITE)
    card(s,M,2.6,5.9,2.2,INK_CARD); put(s,M+0.22,2.82,5.5,0.4,'浅色卡片（灰底黑字）',12,INK)
    card(s,M+6.2,2.6,5.9,2.2,IKB); put(s,M+6.42,2.82,5.5,0.4,'IKB 强调卡（蓝底白字）',12,WHITE)
    footnote(s,'深色页脚注版式 · 白色 62% 透明近似',dark=True)
    # 4 IKB 分屏收尾版式
    p15(prs)
    # 5 样式规范页
    s = blank(prs); bg(s, PAPER); header(s,'样式规范 · STYLE GUIDE','MASTER / SPEC')
    title(s, M, 0.9, '视觉规范', 26)
    sw=[('IKB 克莱因蓝','#002FA7',IKB),('墨黑 Ink','#0A0A0A',INK),('纸白 Paper','#FAFAF8',PAPER),
        ('浅灰卡','#F5F5F4',INK_CARD),('中灰线','#D4D4D2',GREY2),('辅助灰','#737373',GREY3)]
    for i,(n,hx,c) in enumerate(sw):
        x=M+i*2.06; rect(s,x,2.0,1.8,1.0,fill=c,line=GREY2,line_w=0.5)
        put(s,x,3.1,1.9,0.3,n,9.5,INK); put(s,x,3.35,1.9,0.3,hx,8.5,GREY3,latin=SANS)
    put(s,M,4.2,12,0.4,'字体：中文 微软雅黑 / 西文·数字 Arial（无衬线）。标题大字号不加粗，模拟瑞士风细体。',11,INK,lh=1.3)
    put(s,M,4.8,12,0.4,'字号阶梯：封面 40pt · 章节标题 24–27pt · KPI 数字 22–30pt · 正文 10–11pt · 脚注/标签 9pt',10.5,RGBColor(0x44,0x44,0x44),lh=1.3)
    put(s,M,5.4,12,0.4,'版式：直角纯色块、发丝线分隔、单一 IKB 强调色、大幅留白、深/浅/蓝三种页底交替。',10.5,RGBColor(0x44,0x44,0x44),lh=1.3)
    put(s,M,6.0,12,0.4,'明暗节奏：封面(蓝) 浅 深 浅 浅 深 浅 深 浅 浅 深 浅 浅 深 收尾(蓝)',10,GREY3,latin=SANS,lh=1.3)
    prs.save('母版模板.pptx')
    print('母版:', len(prs.slides._sldIdLst), '页 → 母版模板.pptx')

if __name__ == '__main__':
    build_master()
    build_deck()
    print('OK')

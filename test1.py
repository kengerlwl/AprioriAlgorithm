import json
from pyecharts import options as opts
from pyecharts.charts import Graph
from pyecharts.globals import CurrentConfig, NotebookType
CurrentConfig.NOTEBOOK_TYPE = NotebookType.JUPYTER_NOTEBOOK


with open(r'web.json', 'r',encoding='utf_8_sig') as f:
    data = json.load(f)

nodes = [
    opts.GraphNode(
        name= node['name'],
        symbol_size=10, # 节点的显示大小
        value=20, #节点值，点击节点就会显示
        category=node['category'], # 种类
        draggable=True,
    )
    for node in data['nodes']
]


links = [
    opts.GraphLink(
        source=edge['source'],
        target=edge['target'],
    )
    for edge in data['links']
]

categories = [
    {
        'name': category['name']
    }
    for category in data['categories']
]

G = Graph()
G.add(
    series_name='',
    nodes=nodes,
    links=links,
    categories=categories,
    repulsion=30,
    label_opts=opts.LabelOpts(is_show=False),
    linestyle_opts=opts.LineStyleOpts(curve=0.2)
)
G.set_global_opts(
    title_opts=opts.TitleOpts(title='微博转发关系图'),
    legend_opts=opts.LegendOpts(is_show=False)
)
G.render_notebook()

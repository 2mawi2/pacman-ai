import plotly
from plotly import graph_objs as go


def plot_data(statistics):
    xy_data = go.Scatter(x=statistics.x, y=statistics.y, mode='markers',
                         marker=dict(size=8), name='reward', yaxis="y1")
    mov_avg = go.Scatter(x=statistics.x[5:-8], y=statistics.mean_average[5:-8],
                         line=dict(width=2, color='red'), name='Moving reward average', yaxis="y1")
    data = [xy_data, mov_avg]

    plotly.plotly.iplot(data, filename='results')
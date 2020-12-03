from nwbwidgets.timeseries import SeparateTracesPlotlyWidget, show_timeseries


def custom_timeseries_widget_for_behavior(node, **kwargs):
    """Use a custom TimeSeries widget for behavior data"""
    if node.name == 'Velocity':
        return SeparateTracesPlotlyWidget(node)
    else:
        return show_timeseries(node)

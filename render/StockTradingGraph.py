
import dateutil.parser
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import style

# finance module is no longer part of matplotlib
# see: https://github.com/matplotlib/mpl_finance
from mpl_finance import candlestick_ochl as candlestick

style.use('dark_background')

VOLUME_CHART_HEIGHT = 0.33

UP_COLOR = '#27A59A'
DOWN_COLOR = '#EF534F'
UP_TEXT_COLOR = '#73D3CC'
DOWN_TEXT_COLOR = '#DC2C27'


def date2num(date, to_dt=False):
    date = dateutil.parser.parse(str(date))
    if to_dt:
        return date
    return date.timestamp()


class StockTradingGraph:
    """A stock trading visualization using matplotlib made to render OpenAI gym environments"""

    def __init__(self, df, title=None):
        self.df = df
        self.net_worths = np.zeros(len(df['timestamp']))
        self.values = np.zeros(len(df['timestamp']))

        # Create a figure on screen and set the title
        fig = plt.figure()
        fig.suptitle(title)

        # Create top subplot for net worth axis
        self.net_worth_ax = plt.subplot2grid(
            (10, 1), (0, 0), rowspan=2, colspan=1)

        # Create middle subplot for shared price/volume axis
        self.price_ax = plt.subplot2grid(
            (10, 1), (2, 0), rowspan=6, colspan=1, sharex=self.net_worth_ax)

        # Create a new axis for volume which shares its x-axis with price
        self.volume_ax = self.price_ax.twinx()

        # Create bottom subplot for additional info
        self.extra_info_ax = plt.subplot2grid(
            (10, 1), (8, 0), rowspan=2, colspan=1, sharex=self.net_worth_ax)

        # Add padding to make graph easier to view
        plt.subplots_adjust(left=0.11, bottom=0.24,
                            right=0.90, top=0.90, wspace=0.2, hspace=0)

        # Show the graph without blocking the rest of the program
        plt.ion()
        plt.show()

    def _render_extra_info(self, current_step, value, step_range, dates):    # Clear the frame rendered last step
        self.extra_info_ax.clear()

        # Plot net worths
        self.extra_info_ax.plot_date(
            dates, self.values[step_range], '-', label='Indicator')

        # Show legend, which uses the label we defined for the plot above
        last_date = date2num(self.df['timestamp'].values[current_step])
        last_value = self.values[current_step]

        # Annotate the current net worth on the net worth graph
        self.extra_info_ax.annotate('{0:.2f}'.format(value), (last_date, last_value),
                                   xytext=(last_date, last_value),
                                   bbox=dict(boxstyle='round',
                                             fc='w', ec='k', lw=1),
                                   color="black",
                                   fontsize="small")

        # Add space above and below min/max net worth
        if len(self.values[np.nonzero(self.values)]) > 0:
            self.extra_info_ax.set_ylim(
                min(self.values[np.nonzero(self.values)]) / 1.25, max(self.values) * 1.25)


    def _render_net_worth(self, current_step, net_worth, step_range, dates):
        # Clear the frame rendered last step
        self.net_worth_ax.clear()

        # Plot net worths
        self.net_worth_ax.plot_date(
            dates, self.net_worths[step_range], '-', label='Net Worth')

        # Show legend, which uses the label we defined for the plot above
        self.net_worth_ax.legend()
        legend = self.net_worth_ax.legend(loc=2, ncol=2, prop={'size': 8})
        legend.get_frame().set_alpha(0.4)

        last_date = date2num(self.df['timestamp'].values[current_step])
        last_net_worth = self.net_worths[current_step]

        # Annotate the current net worth on the net worth graph
        self.net_worth_ax.annotate('{0:.2f}'.format(net_worth), (last_date, last_net_worth),
                                   xytext=(last_date, last_net_worth),
                                   bbox=dict(boxstyle='round',
                                             fc='w', ec='k', lw=1),
                                   color="black",
                                   fontsize="small")

        # Add space above and below min/max net worth
        self.net_worth_ax.set_ylim(
            min(self.net_worths[np.nonzero(self.net_worths)]) / 1.25, max(self.net_worths) * 1.25)

    def _render_price(self, current_step, net_worth, dates, step_range):
        self.price_ax.clear()

        # Format data for OHCL candlestick graph
        candlesticks = zip(dates,
                           self.df['open'].values[step_range], self.df['close'].values[step_range],
                           self.df['high'].values[step_range], self.df['low'].values[step_range])

        # Plot price using candlestick graph from mpl_finance
        candlestick(self.price_ax, candlesticks, width=1,
                    colorup=UP_COLOR, colordown=DOWN_COLOR)

        last_date = date2num(self.df['timestamp'].values[current_step])
        last_close = self.df['close'].values[current_step]
        last_high = self.df['high'].values[current_step]

        # Print the current price to the price axis
        self.price_ax.annotate('{0:.2f}'.format(last_close), (last_date, last_close),
                               xytext=(last_date, last_high),
                               bbox=dict(boxstyle='round',
                                         fc='w', ec='k', lw=1),
                               color="black",
                               fontsize="small")

        # Shift price axis up to give volume chart space
        ylim = self.price_ax.get_ylim()
        self.price_ax.set_ylim(ylim[0] - (ylim[1] - ylim[0])
                               * VOLUME_CHART_HEIGHT, ylim[1])

    def _render_volume(self, current_step, net_worth, dates, step_range):
        self.volume_ax.clear()

        volume = np.array(self.df['volume'].values[step_range])

        pos = self.df['open'].values[step_range] - \
            self.df['close'].values[step_range] < 0
        neg = self.df['open'].values[step_range] - \
            self.df['close'].values[step_range] > 0

        # Color volume bars based on price direction on that date
        self.volume_ax.bar(dates[pos], volume[pos], color=UP_COLOR,
                           alpha=0.4, width=1, align='center')
        self.volume_ax.bar(dates[neg], volume[neg], color=DOWN_COLOR,
                           alpha=0.4, width=1, align='center')

        # Cap volume axis height below price chart and hide ticks
        self.volume_ax.set_ylim(0, max(volume) / VOLUME_CHART_HEIGHT)
        self.volume_ax.yaxis.set_ticks([])

    def _render_trades(self, current_step, trades, step_range):
        for trade in trades:
            if trade['step'] in step_range:
                date = date2num(self.df['timestamp'].values[trade['step']])
                high = self.df['high'].values[trade['step']]
                low = self.df['low'].values[trade['step']]

                if trade['type'] == 'buy':
                    high_low = low
                    color = UP_TEXT_COLOR
                else:
                    high_low = high
                    color = DOWN_TEXT_COLOR

                total = '{0:.2f}'.format(trade['total'])

                # Print the current price to the price axis
                self.price_ax.annotate(f'${total}', (date, high_low),
                                       xytext=(date, high_low),
                                       color=color,
                                       fontsize=8,
                                       arrowprops=(dict(color=color)))

    def render(self, current_step, net_worth, trades, window_size=40, extra_value=0):
        self.net_worths[current_step] = net_worth
        self.values[current_step] = extra_value

        window_start = max(current_step - window_size, 0)
        step_range = range(window_start, current_step + 1)

        # Format dates as timestamps, necessary for candlestick graph
        dates = np.array([date2num(x)
                          for x in self.df['timestamp'].values[step_range]])

        self._render_net_worth(current_step, net_worth, step_range, dates)
        self._render_extra_info(current_step, extra_value, step_range, dates)
        self._render_price(current_step, net_worth, dates, step_range)
        self._render_volume(current_step, net_worth, dates, step_range)
        self._render_trades(current_step, trades, step_range)

        # Format the date ticks to be more easily read
        self.extra_info_ax.set_xticklabels(self.df['timestamp'].values[step_range], rotation=45,
                                      horizontalalignment='right')

        # Hide duplicate net worth date labels
        plt.setp(self.net_worth_ax.get_xticklabels(), visible=False)
        plt.setp(self.price_ax.get_xticklabels(), visible=False)

        # Necessary to view frames before they are unrendered
        plt.pause(0.001)

    def close(self):
        plt.close()

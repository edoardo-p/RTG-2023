# Copyright 2021 Optiver Asia Pacific Pty. Ltd.
#
# This file is part of Ready Trader Go.
#
#     Ready Trader Go is free software: you can redistribute it and/or
#     modify it under the terms of the GNU Affero General Public License
#     as published by the Free Software Foundation, either version 3 of
#     the License, or (at your option) any later version.
#
#     Ready Trader Go is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Affero General Public License for more details.
#
#     You should have received a copy of the GNU Affero General Public
#     License along with Ready Trader Go.  If not, see
#     <https://www.gnu.org/licenses/>.
import asyncio
import itertools
from typing import List

import numpy as np
import pandas as pd

from ready_trader_go import (
    MAXIMUM_ASK,
    MINIMUM_BID,
    BaseAutoTrader,
    Instrument,
    Lifespan,
    Side,
)

LOT_SIZE = 10
POSITION_LIMIT = 100
TICK_SIZE_IN_CENTS = 100
MIN_BID_NEAREST_TICK = (
    (MINIMUM_BID + TICK_SIZE_IN_CENTS) // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS
)
MAX_ASK_NEAREST_TICK = MAXIMUM_ASK // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS


def ewma(arr: np.ndarray, window: int, alpha: float) -> np.ndarray:
    w0 = np.full((window, window), 1 - alpha)
    p = np.vstack([np.arange(i, i - window, -1) for i in range(window)])
    w = np.tril(w0**p, 0)
    return np.dot(w, arr[:: np.newaxis]) / w.sum(axis=1)


def emas(
    average: np.ndarray, n_fast=8, n_medium=13, n_slow=21
) -> tuple[float, float, float]:
    ema_fast = ewma(average, n_fast, 2 / (n_fast + 1))
    ema_medium = ewma(average, n_medium, 2 / (n_medium + 1))
    ema_slow = ewma(average, n_slow, 2 / (n_slow + 1))
    return ema_fast[-1], ema_medium[-1], ema_slow[-1]


def macd_indicators(
    average: np.ndarray, n_fast=12, n_slow=26, n_signal=9
) -> tuple[float, float, float]:
    ema_fast = ewma(average, n_fast, 2 / (n_fast + 1))
    ema_slow = ewma(average, n_slow, 2 / (n_slow + 1))
    macd = ema_fast - ema_slow
    signal_line = ewma(macd, n_signal, 2 / (n_signal + 1))
    histogram = macd - signal_line
    # print(macd[-1], signal_line[-1])
    return macd[-1], signal_line[-1], histogram[-1]


def rsi(df: pd.DataFrame, n=14) -> tuple[float]:
    prices = (df["bid"] + df["ask"]) / 2
    deltas = np.diff(prices)
    seed = deltas[: n + 1]
    up = seed[seed >= 0].sum() / n
    down = -seed[seed < 0].sum() / n
    rs = up / down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100.0 - 100.0 / (1.0 + rs)

    for i in range(n, len(prices)):
        delta = deltas[i - 1]  # cause the diff is 1 shorter

        if delta > 0:
            upval = delta
            downval = 0.0
        else:
            upval = 0.0
            downval = -delta

        up = (up * (n - 1) + upval) / n
        down = (down * (n - 1) + downval) / n

        rs = up / down
        rsi[i] = 100.0 - 100.0 / (1.0 + rs)

    return rsi[-1]


def rollover(arr: np.ndarray, new_val: float) -> np.ndarray:
    new_arr = np.roll(arr, -1)
    new_arr[-1] = new_val
    return new_arr


class AutoTrader(BaseAutoTrader):
    """
    Implements the MACD Zero Crosses indicator with RSI filtering.+
    When the MACD is greater than zero and the RSI is greater than 60%, we open a long position,
    when the MACD is less than zero and the RSI is less than 40%, we open a short position.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, team_name: str, secret: str):
        """Initialise a new instance of the AutoTrader class."""
        super().__init__(loop, team_name, secret)
        self.order_ids = itertools.count(1)
        self.bids = set()
        self.asks = set()
        self.ask_id = self.ask_price = self.bid_id = self.bid_price = self.position = 0

        self.max_window = 26
        self.etf_prices = np.zeros((self.max_window,))

        self.long_position_opened = False
        self.short_position_opened = False

    def on_error_message(self, client_order_id: int, error_message: bytes) -> None:
        """Called when the exchange detects an error.

        If the error pertains to a particular order, then the client_order_id
        will identify that order, otherwise the client_order_id will be zero.
        """
        self.logger.warning(
            f"error with order {client_order_id}: {error_message.decode()}"
        )
        if client_order_id != 0 and (
            client_order_id in self.bids or client_order_id in self.asks
        ):
            self.on_order_status_message(client_order_id, 0, 0, 0)

    def on_hedge_filled_message(
        self, client_order_id: int, price: int, volume: int
    ) -> None:
        """Called when one of your hedge orders is filled.

        The price is the average price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.
        """
        self.logger.info(
            f"received hedge filled for order {client_order_id} with average price {price} and volume {volume}"
        )

    def on_order_book_update_message(
        self,
        instrument: int,
        sequence_number: int,
        ask_prices: List[int],
        ask_volumes: List[int],
        bid_prices: List[int],
        bid_volumes: List[int],
    ) -> None:
        """Called periodically to report the status of an order book.

        The sequence number can be used to detect missed or out-of-order
        messages. The five best available ask (i.e. sell) and bid (i.e. buy)
        prices are reported along with the volume available at each of those
        price levels.
        """
        self.logger.info(
            f"received order book for instrument {instrument} with sequence number {sequence_number}"
        )

        if instrument == Instrument.ETF:
            self.etf_prices = rollover(
                self.etf_prices, (bid_prices[0] + ask_prices[0] / 2)
            )

        if len(self.etf_prices) >= self.max_window:
            macd, signal_line, _ = macd_indicators(self.etf_prices)
            # ema_fast, ema_medium, ema_slow = emas(self.etf_prices)
            # rsi = self.rsi(self.df_etf)
            new_bid_price, new_ask_price = bid_prices[0], ask_prices[0]

            if self.bid_id != 0 and new_bid_price not in (self.bid_price, 0):
                self.send_cancel_order(self.bid_id)
                self.bid_id = 0
                return

            if self.ask_id != 0 and new_ask_price not in (self.ask_price, 0):
                self.send_cancel_order(self.ask_id)
                self.ask_id = 0
                return

            # if (
            #     ema_fast < ema_medium
            #     and ema_medium < ema_slow
            #     and macd < signal_line
            #     and macd < 0
            #     and self.short_position_opened == False
            #     and self.ask_id == 0
            #     and new_ask_price != 0
            #     and self.position > LOT_SIZE - POSITION_LIMIT
            # ):
            #     self.ask_id = next(self.order_ids)
            #     self.ask_price = new_ask_price
            #     self.send_insert_order(
            #         self.ask_id,
            #         Side.SELL,
            #         new_ask_price,
            #         LOT_SIZE,
            #         Lifespan.GOOD_FOR_DAY,
            #     )
            #     self.asks.add(self.ask_id)

            # Open long
            if (
                macd > signal_line
                and macd < 0
                and not self.long_position_opened
                and self.bid_id == 0
                and new_bid_price != 0
                and self.position < POSITION_LIMIT - LOT_SIZE
            ):
                self.bid_id = next(self.order_ids)
                self.bid_price = new_bid_price
                self.send_insert_order(
                    self.bid_id,
                    Side.BUY,
                    new_bid_price,
                    abs(LOT_SIZE - self.position),
                    Lifespan.GOOD_FOR_DAY,
                )
                self.bids.add(self.bid_id)

            # Close long
            if (
                macd < signal_line
                and macd > 0
                and self.long_position_opened
                and self.ask_id == 0
                and new_ask_price != 0
            ):
                self.ask_id = next(self.order_ids)
                self.ask_price = new_ask_price
                self.send_insert_order(
                    self.ask_id,
                    Side.SELL,
                    new_ask_price,
                    abs(self.position),
                    Lifespan.GOOD_FOR_DAY,
                )
                self.asks.add(self.ask_id)

    def on_order_filled_message(
        self, client_order_id: int, price: int, volume: int
    ) -> None:
        """Called when one of your orders is filled, partially or fully.

        The price is the price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.
        """
        self.logger.info(
            f"received order filled for order {client_order_id} with price {price} and volume {volume}"
        )
        if client_order_id in self.bids:
            # self.short_position_opened = False
            self.position += volume
            if self.position >= LOT_SIZE:
                self.long_position_opened = True
            # self.send_hedge_order(
            #     next(self.order_ids), Side.ASK, MIN_BID_NEAREST_TICK, volume
            # )
        elif client_order_id in self.asks:
            # self.short_position_opened = True
            self.position -= volume
            if self.position < 0:
                self.long_position_opened = False
            # self.send_hedge_order(
            #     next(self.order_ids), Side.BID, MAX_ASK_NEAREST_TICK, volume
            # )

    def on_order_status_message(
        self, client_order_id: int, fill_volume: int, remaining_volume: int, fees: int
    ) -> None:
        """Called when the status of one of your orders changes.

        The fill_volume is the number of lots already traded, remaining_volume
        is the number of lots yet to be traded and fees is the total fees for
        this order. Remember that you pay fees for being a market taker, but
        you receive fees for being a market maker, so fees can be negative.

        If an order is cancelled its remaining volume will be zero.
        """
        self.logger.info(
            f"received order status for order {client_order_id} with fill volume {fill_volume} remaining {remaining_volume} and fees {fees}"
        )
        if remaining_volume == 0:
            if client_order_id == self.bid_id:
                self.bid_id = 0
                self.bids.discard(client_order_id)
            elif client_order_id == self.ask_id:
                self.ask_id = 0
                self.asks.discard(client_order_id)

    def on_trade_ticks_message(
        self,
        instrument: int,
        sequence_number: int,
        ask_prices: List[int],
        ask_volumes: List[int],
        bid_prices: List[int],
        bid_volumes: List[int],
    ) -> None:
        """Called periodically when there is trading activity on the market.

        The five best ask (i.e. sell) and bid (i.e. buy) prices at which there
        has been trading activity are reported along with the aggregated volume
        traded at each of those price levels.

        If there are less than five prices on a side, then zeros will appear at
        the end of both the prices and volumes arrays.
        """
        self.logger.info(
            f"received trade ticks for instrument {instrument} with sequence number {sequence_number}"
        )

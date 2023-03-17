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


def calc_ema(df, period, alpha=False):

    con = pd.concat(
        [df[:period]["TR"].rolling(window=period).mean(), df[period:]["TR"]]
    )

    if alpha == True:
        # (1 - alpha) * previous_val + alpha * current_val where alpha = 1 / period
        alpha = 1 / period
        target = con.ewm(alpha, adjust=False).mean()
    else:
        # ((current_val - previous_val) * coeff) + previous_val where coeff = 2 / (period + 1)
        target = con.ewm(span=period, adjust=False).mean()

    target.fillna(0, inplace=True)
    return target


def calc_atr(df, period):
    # Compute true range only if it is not computed and stored earlier in the df
    df["h-l"] = df["High"] - df["Low"]
    df["h-yc"] = abs(df["High"] - df["Close"].shift())
    df["l-yc"] = abs(df["Low"] - df["Close"].shift())

    df["TR"] = df[["h-l", "h-yc", "l-yc"]].max(axis=1)

    df.drop(["h-l", "h-yc", "l-yc"], inplace=True, axis=1)

    # Compute EMA of true range using ATR formula after ignoring first row
    df["ATR"] = calc_ema(df, period, alpha=True)

    return df


def calc_supertrend(df, period, multiplier):

    calc_atr(df, period)
    atr = f"ATR_{period}"
    st = f"ST_{period}_{multiplier}"
    stx = f"STX_{period}_{multiplier}"

    """
    SuperTrend Algorithm :
    
        BASIC UPPERBAND = (HIGH + LOW) / 2 + Multiplier * ATR
        BASIC LOWERBAND = (HIGH + LOW) / 2 - Multiplier * ATR
        
        FINAL UPPERBAND = IF( (Current BASICUPPERBAND < Previous FINAL UPPERBAND) or (Previous Close > Previous FINAL UPPERBAND))
                            THEN (Current BASIC UPPERBAND) ELSE Previous FINALUPPERBAND)
        FINAL LOWERBAND = IF( (Current BASIC LOWERBAND > Previous FINAL LOWERBAND) or (Previous Close < Previous FINAL LOWERBAND)) 
                            THEN (Current BASIC LOWERBAND) ELSE Previous FINAL LOWERBAND)
        
        SUPERTREND = IF((Previous SUPERTREND = Previous FINAL UPPERBAND) and (Current Close <= Current FINAL UPPERBAND)) THEN
                        Current FINAL UPPERBAND
                    ELSE
                        IF((Previous SUPERTREND = Previous FINAL UPPERBAND) and (Current Close > Current FINAL UPPERBAND)) THEN
                            Current FINAL LOWERBAND
                        ELSE
                            IF((Previous SUPERTREND = Previous FINAL LOWERBAND) and (Current Close >= Current FINAL LOWERBAND)) THEN
                                Current FINAL LOWERBAND
                            ELSE
                                IF((Previous SUPERTREND = Previous FINAL LOWERBAND) and (Current Close < Current FINAL LOWERBAND)) THEN
                                    Current FINAL UPPERBAND
    """

    # Compute basic upper and lower bands
    df["basic_ub"] = (df["High"] + df["Low"]) / 2 + multiplier * df[atr]
    df["basic_lb"] = (df["High"] + df["Low"]) / 2 - multiplier * df[atr]

    # Compute final upper and lower bands
    df["final_ub"] = 0.00
    df["final_lb"] = 0.00
    for i in range(period, len(df)):
        df["final_ub"].iat[i] = (
            df["basic_ub"].iat[i]
            if df["basic_ub"].iat[i] < df["final_ub"].iat[i - 1]
            or df["Close"].iat[i - 1] > df["final_ub"].iat[i - 1]
            else df["final_ub"].iat[i - 1]
        )
        df["final_lb"].iat[i] = (
            df["basic_lb"].iat[i]
            if df["basic_lb"].iat[i] > df["final_lb"].iat[i - 1]
            or df["Close"].iat[i - 1] < df["final_lb"].iat[i - 1]
            else df["final_lb"].iat[i - 1]
        )

    # Set the Supertrend value
    df[st] = 0.00
    for i in range(period, len(df)):
        df[st].iat[i] = (
            df["final_ub"].iat[i]
            if df[st].iat[i - 1] == df["final_ub"].iat[i - 1]
            and df["Close"].iat[i] <= df["final_ub"].iat[i]
            else df["final_lb"].iat[i]
            if df[st].iat[i - 1] == df["final_ub"].iat[i - 1]
            and df["Close"].iat[i] > df["final_ub"].iat[i]
            else df["final_lb"].iat[i]
            if df[st].iat[i - 1] == df["final_lb"].iat[i - 1]
            and df["Close"].iat[i] >= df["final_lb"].iat[i]
            else df["final_ub"].iat[i]
            if df[st].iat[i - 1] == df["final_lb"].iat[i - 1]
            and df["Close"].iat[i] < df["final_lb"].iat[i]
            else 0.00
        )

    # Mark the trend direction up/down
    df[stx] = np.where(
        (df[st] > 0.00), np.where((df["Close"] < df[st]), "down", "up"), np.NaN
    )

    # Remove basic and final bands from the columns
    df.drop(["basic_ub", "basic_lb", "final_ub", "final_lb"], inplace=True, axis=1)

    df.fillna(0, inplace=True)

    return df


class AutoTrader(BaseAutoTrader):
    """Example Auto-trader.

    When it starts this auto-trader places ten-lot bid and ask orders at the
    current best-bid and best-ask prices respectively. Thereafter, if it has
    a long position (it has bought more lots than it has sold) it reduces its
    bid and ask prices. Conversely, if it has a short position (it has sold
    more lots than it has bought) then it increases its bid and ask prices.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, team_name: str, secret: str):
        """Initialise a new instance of the AutoTrader class."""
        super().__init__(loop, team_name, secret)
        self.order_ids = itertools.count(1)
        self.bids = set()
        self.asks = set()
        self.ask_id = self.ask_price = self.bid_id = self.bid_price = self.position = 0

    def on_error_message(self, client_order_id: int, error_message: bytes) -> None:
        """Called when the exchange detects an error.

        If the error pertains to a particular order, then the client_order_id
        will identify that order, otherwise the client_order_id will be zero.
        """
        self.logger.warning(
            "error with order %d: %s", client_order_id, error_message.decode()
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
            "received hedge filled for order %d with average price %d and volume %d",
            client_order_id,
            price,
            volume,
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
            "received order book for instrument %d with sequence number %d",
            instrument,
            sequence_number,
        )

        if bid_volumes[0] | ask_volumes[0] == 0:
            return
        new_bid_price, new_ask_price = bid_prices[0], ask_prices[0]

        if self.bid_id != 0 and new_bid_price not in (self.bid_price, 0):
            self.send_cancel_order(self.bid_id)
            self.bid_id = 0

        if self.ask_id != 0 and new_ask_price not in (self.ask_price, 0):
            self.send_cancel_order(self.ask_id)
            self.ask_id = 0

        if (
            self.ask_id == 0
            and new_ask_price != 0
            and self.position > LOT_SIZE - POSITION_LIMIT
        ):
            self.ask_id = next(self.order_ids)
            self.ask_price = new_ask_price
            self.send_insert_order(
                self.ask_id,
                Side.SELL,
                new_ask_price,
                LOT_SIZE,
                Lifespan.FILL_AND_KILL,
            )
            self.asks.add(self.ask_id)

        elif (
            self.bid_id == 0
            and new_bid_price != 0
            and self.position < POSITION_LIMIT - LOT_SIZE
        ):
            self.bid_id = next(self.order_ids)
            self.bid_price = new_bid_price
            self.send_insert_order(
                self.bid_id,
                Side.BUY,
                new_bid_price,
                LOT_SIZE,
                Lifespan.FILL_AND_KILL,
            )
            self.bids.add(self.bid_id)

    def on_order_filled_message(
        self, client_order_id: int, price: int, volume: int
    ) -> None:
        """Called when one of your orders is filled, partially or fully.

        The price is the price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.
        """
        self.logger.info(
            "received order filled for order %d with price %d and volume %d",
            client_order_id,
            price,
            volume,
        )
        if client_order_id in self.bids:
            self.position += volume
            self.send_hedge_order(
                next(self.order_ids), Side.ASK, MIN_BID_NEAREST_TICK, volume
            )
        elif client_order_id in self.asks:
            self.position -= volume
            self.send_hedge_order(
                next(self.order_ids), Side.BID, MAX_ASK_NEAREST_TICK, volume
            )

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
            "received order status for order %d with fill volume %d remaining %d and fees %d",
            client_order_id,
            fill_volume,
            remaining_volume,
            fees,
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
            "received trade ticks for instrument %d with sequence number %d",
            instrument,
            sequence_number,
        )

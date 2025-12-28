#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 27 16:22:14 2025

@author: alex
"""

from datetime import datetime
from collections import defaultdict


class Account:
    def __init__(self, name):
        self.name = name
        self.transactions = []

    @property
    def balance(self):
        return sum(t.amount for t in self.transactions)

    def balance_a_fecha(self, fecha):
        return sum(t.amount for t in self.transactions if t.timestamp <= fecha)

    def __repr__(self):
        return f"<Account: {self.name}, Balance: {self.balance:.2f}>"


class Transaction:
    def __init__(self, amount, description, category, account, timestamp=None):
        self.amount = amount
        self.description = description
        self.category = category
        self.account = account
        self.timestamp = timestamp if timestamp else datetime.now()

    def __repr__(self):
        return f"<Transaction {self.timestamp.date()} | {self.amount:.2f} | {self.category} | {self.account.name}>"


class Budget:
    def __init__(self):
        self.accounts = {}
        self.transactions = []

    def add_account(self, account):
        self.accounts[account.name] = account

    def get_account(self, name):
        return self.accounts.get(name)

    def add_transaction(self, transaction):
        if transaction.account.name not in self.accounts:
            raise ValueError(f"La cuenta '{transaction.account.name}' no está en el presupuesto.")
        self.accounts[transaction.account.name].transactions.append(transaction)
        self.transactions.append(transaction)

    def get_balance(self):
        return sum(account.balance for account in self.accounts.values())

    def balances_a_fecha(self, fecha):
        return {nombre: cuenta.balance_a_fecha(fecha) for nombre, cuenta in self.accounts.items()}

    def balance_total_a_fecha(self, fecha):
        return sum(cuenta.balance_a_fecha(fecha) for cuenta in self.accounts.values())

    def get_transactions(self, category=None, account=None):
        result = self.transactions
        if category:
            result = [t for t in result if t.category == category]
        if account:
            result = [t for t in result if t.account.name == account]
        return result

    def get_transactions_by_month(self):
        months = defaultdict(list)
        for t in self.transactions:
            key = t.timestamp.strftime("%Y-%m")
            months[key].append(t)
        return months

    def __repr__(self):
        return f"<Budget: {len(self.accounts)} cuentas, {len(self.transactions)} transacciones>"

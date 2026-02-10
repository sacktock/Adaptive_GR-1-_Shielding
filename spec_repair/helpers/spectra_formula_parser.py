from py_ltl.formula import (
    AtomicProposition, Not, And, Or, Implies, Until,
    Next, Prev, Globally, Eventually, Top, Bottom
)
import re


class Token:
    def __init__(self, type_, value):
        self.type = type_
        self.value = value

    def __repr__(self):
        return f"Token({self.type}, {self.value})"


class LTLTokenizer:
    token_spec = [
        ('IMPLIES', r'->'),
        ('UNTIL', r'\bU\b'),
        ('NEXT', r'\bX\b|\bnext\b'),
        ('PREV', r'\bprev\b|\bPREV\b'),
        ('GLOBALLY', r'\bG\b|\balw\b'),
        ('EVENTUALLY', r'\bF\b'),
        ('TRUE', r'\btrue\b'),
        ('FALSE', r'\bfalse\b'),
        ('AND', r'&'),
        ('OR', r'\|'),
        ('NOT', r'!'),
        ('EQ', r'='),
        ('LPAREN', r'\('),
        ('RPAREN', r'\)'),
        ('SEMI', r';'),
        ('ID', r'[A-Za-z_][A-Za-z0-9_]*'),
        ('SKIP', r'[ \t]+'),
    ]
    tok_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_spec)

    def __init__(self, text):
        self.tokens = list(self.generate_tokens(text))
        self.pos = 0

    def generate_tokens(self, text):
        pos = 0
        while pos < len(text):
            match = re.match(self.tok_regex, text[pos:])
            if not match:
                raise ValueError(f"Unexpected character: {text[pos]}")
            kind = match.lastgroup
            value = match.group()
            if kind == 'SKIP':
                pos += len(value)
                continue

            # Special case: split compound prefix ops like GF, alwEv, etc.
            if kind == 'ID':
                remaining = text[pos + len(value):].lstrip()
                if remaining.startswith("("):
                    prefixes = {
                        "G": "GLOBALLY",
                        "F": "EVENTUALLY",
                        "X": "NEXT",
                        "alw": "GLOBALLY",
                        "Ev": "EVENTUALLY",
                    }

                    i = 0
                    matched = True
                    while i < len(value):
                        for p, t in prefixes.items():
                            if value.startswith(p, i):
                                yield Token(t, p)
                                i += len(p)
                                break
                        else:
                            matched = False
                            break

                    if matched:
                        pos += len(value)
                        continue

            yield Token(kind, value)
            pos += len(value)

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self, expected_type=None):
        tok = self.peek()
        if not tok:
            raise ValueError("Unexpected end of input")
        if expected_type and tok.type != expected_type:
            raise ValueError(f"Expected {expected_type}, got {tok.type}")
        self.pos += 1
        return tok


class SpectraFormulaParser:
    def parse(self, text: str):
        self.lexer = LTLTokenizer(text)
        expr = self.expression()
        if self.lexer.peek() and self.lexer.peek().type == "SEMI":
            self.lexer.consume("SEMI")
        if self.lexer.peek():
            raise ValueError(f"Unexpected trailing token: {self.lexer.peek()}")
        return expr

    def expression(self, min_prec=0):
        left = self.atom()

        while True:
            tok = self.lexer.peek()
            if not tok:
                break

            entry = self.infix_operators.get(tok.type)
            if not entry:
                break

            prec, assoc, handler = entry
            if prec < min_prec:
                break

            self.lexer.consume()
            next_min_prec = prec + 1 if assoc == 'left' else prec
            right = self.expression(next_min_prec)
            left = handler(left, right)

        return left

    def atom(self):
        tok = self.lexer.peek()
        if not tok:
            raise ValueError("Unexpected end of input in atom")

        if tok.type == 'LPAREN':
            self.lexer.consume('LPAREN')
            expr = self.expression()
            self.lexer.consume('RPAREN')
            return expr
        elif tok.type == 'NOT':
            self.lexer.consume()
            return Not(self.atom())
        elif tok.type == 'NEXT':
            self.lexer.consume()
            return Next(self.atom())
        elif tok.type == 'PREV':
            self.lexer.consume()
            return Prev(self.atom())
        elif tok.type == 'GLOBALLY':
            self.lexer.consume()
            return Globally(self.atom())
        elif tok.type == 'EVENTUALLY':
            self.lexer.consume()
            return Eventually(self.atom())
        elif tok.type == 'TRUE':
            self.lexer.consume()
            return Top()
        elif tok.type == 'FALSE':
            self.lexer.consume()
            return Bottom()
        elif tok.type == 'ID':
            id_token = self.lexer.consume()
            next_tok = self.lexer.peek()
            if next_tok and next_tok.type == 'EQ':
                self.lexer.consume('EQ')
                val_tok = self.lexer.consume()
                value = self._parse_value(val_tok.value)
                return AtomicProposition(id_token.value, value)
            else:
                return AtomicProposition(id_token.value, True)
        else:
            raise ValueError(f"Unexpected token in atom: {tok.type}")

    def _parse_value(self, val: str):
        if val.lower() == 'true':
            return True
        if val.lower() == 'false':
            return False
        if val.isdigit():
            return int(val)
        return val

    infix_operators = {
        'AND': (10, 'left', lambda a, b: And(a, b)),
        'OR': (9, 'left', lambda a, b: Or(a, b)),
        'IMPLIES': (5, 'right', lambda a, b: Implies(a, b)),
        #'UNTIL': (8, 'left', lambda a, b: Until(a, b)),
        'UNTIL': (8, 'left', lambda a, b: (_ for _ in ()).throw(NotImplementedError("Until operator is not supported"))),
    }

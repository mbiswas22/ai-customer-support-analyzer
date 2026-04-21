import sqlite3
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tickets.db")
SERVER_PORT = 8765


def _conn():
    return sqlite3.connect(DB_PATH)


def _rows_to_json(cursor) -> str:
    cols = [d[0] for d in cursor.description]
    return json.dumps([dict(zip(cols, row)) for row in cursor.fetchall()])


class DBHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass  # suppress console noise

    def _respond(self, body: str, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body.encode())

    def do_GET(self):
        parsed = urlparse(self.path)
        params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        path = parsed.path

        try:
            with _conn() as con:
                cur = con.cursor()

                if path == "/tickets":
                    clauses, args = [], []
                    for col in ("category", "sentiment", "priority"):
                        if col in params and params[col] != "All":
                            clauses.append(f"{col} = ?")
                            args.append(params[col])
                    if params.get("urgent_only") == "1":
                        clauses.append("urgent = 1")
                    if "start_date" in params:
                        clauses.append("created_at >= ?")
                        args.append(params["start_date"])
                    if "end_date" in params:
                        clauses.append("created_at <= ?")
                        args.append(params["end_date"])
                    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
                    cur.execute(f"SELECT * FROM tickets {where}", args)
                    self._respond(_rows_to_json(cur))

                elif path == "/tickets/all":
                    cur.execute("SELECT * FROM tickets")
                    self._respond(_rows_to_json(cur))

                elif path == "/tickets/texts":
                    cur.execute("SELECT text FROM tickets")
                    self._respond(json.dumps([r[0] for r in cur.fetchall()]))

                elif path == "/tickets/distinct":
                    col = params.get("column", "category")
                    cur.execute(f"SELECT DISTINCT {col} FROM tickets ORDER BY {col}")
                    self._respond(json.dumps([r[0] for r in cur.fetchall() if r[0]]))

                elif path == "/tickets/date_range":
                    cur.execute("SELECT MIN(created_at), MAX(created_at) FROM tickets")
                    row = cur.fetchone()
                    self._respond(json.dumps({"min": row[0], "max": row[1]}))

                elif path == "/tickets/daily_counts":
                    category = params.get("category")
                    if category and category != "All":
                        cur.execute("""
                            SELECT DATE(created_at) as date, COUNT(*) as count
                            FROM tickets WHERE category = ?
                            GROUP BY DATE(created_at) ORDER BY date
                        """, [category])
                    else:
                        cur.execute("""
                            SELECT DATE(created_at) as date, COUNT(*) as count
                            FROM tickets GROUP BY DATE(created_at) ORDER BY date
                        """)
                    self._respond(_rows_to_json(cur))

                else:
                    self._respond(json.dumps({"error": "Not found"}), 404)

        except Exception as e:
            self._respond(json.dumps({"error": str(e)}), 500)

    def do_POST(self):
        if self.path == "/tickets/load":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            try:
                with _conn() as con:
                    con.execute("DROP TABLE IF EXISTS tickets")
                    if body:
                        cols = list(body[0].keys())
                        placeholders = ", ".join("?" * len(cols))
                        col_defs = ", ".join(f"{c} TEXT" for c in cols)
                        con.execute(f"CREATE TABLE tickets ({col_defs})")
                        con.executemany(
                            f"INSERT INTO tickets ({', '.join(cols)}) VALUES ({placeholders})",
                            [[str(row[c]) for c in cols] for row in body]
                        )
                self._respond(json.dumps({"inserted": len(body)}))
            except Exception as e:
                self._respond(json.dumps({"error": str(e)}), 500)
        else:
            self._respond(json.dumps({"error": "Not found"}), 404)


def start_server():
    server = HTTPServer(("localhost", SERVER_PORT), DBHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return f"http://localhost:{SERVER_PORT}"

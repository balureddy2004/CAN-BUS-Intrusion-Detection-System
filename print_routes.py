# print_routes.py
from server import app
for rule in sorted(app.url_map.iter_rules(), key=lambda r: r.rule):
    methods = ",".join(sorted(rule.methods))
    print(f"{rule.endpoint:40s}  {rule.rule:30s}  {methods}")

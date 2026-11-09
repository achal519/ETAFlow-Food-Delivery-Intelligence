with open('app/index.html', encoding='utf-8') as f:
    html = f.read()

card_ids = [
    'sim-card-order-id', 'sim-card-category-city', 'sim-card-status-pill',
    'sim-card-predicted', 'sim-card-elapsed', 'sim-card-remaining-hint',
    'sim-card-haversine', 'sim-card-progress-pct', 'sim-card-progress-fill',
    'stage-milestone-received', 'stage-milestone-prep', 'stage-milestone-pickup',
    'stage-milestone-transit', 'stage-milestone-delivered', 'sim-stage-explainer-text'
]
for cid in card_ids:
    if f'id="{cid}"' not in html:
        print(f"MISSING in HTML: {cid}")
print("Card IDs check done.")

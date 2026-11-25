import json
s = json.load(open(r"C:\Users\patri\lbf20m-run-20261125\report\summary.json"))


def show(name, blk):
    print("==", name)
    for a, o in blk["arms"].items():
        print(f"{a:16s} ex={o['exact']:2d}/{o['roots']:2d} caps={o['cap_reasons']} base={o['base_bits']:,} qbits={o['query_bits']:,} "
              f"msgs={o['messages']:,} hits={o['query_hits']} miss={o['query_misses']} states={o['diagram_states']} edges={o['diagram_edges']} "
              f"labels={o['compatible_labels']} term={o['terminals']} total_ms={o['total_seconds_sum_of_medians_ms']:.1f} "
              f"setup={o['setup_seconds_sum_of_medians_ms']:.1f} query={o['query_seconds_sum_of_medians_ms']:.1f}")
    for p in blk["paired"]:
        if p["roots"] and p["a"].startswith("mdd") and p["b"] in ("mdd_forest", "mdd_components", "dfs_cover"):
            print(f"  paired {p['a']} vs {p['b']}: roots={p['roots']} ms {p['total_ms_a']:.1f} vs {p['total_ms_b']:.1f} "
                  f"ratio={p['ratio_a_over_b']:.2f} a_faster_on={p['roots_where_a_faster']} bits {p['base_bits_a']:,}+{p['query_bits_a']:,} vs "
                  f"{p['base_bits_b']:,}+{p['query_bits_b']:,} msgs {p['messages_a']} vs {p['messages_b']}")


for k in ("fresh_explosion", "retro_capped", "retro_bipartite", "retro_mixed"):
    show(k, s["cohorts"][k])
show("retro_all_50", s["all_roots"]["retro_all_50"])
print(s["process"])

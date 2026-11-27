**Gate outcome, ground-truth head box (decision protocol).** Δ = relative reduction of the miss rate (needs ≥ 0.10) / NLL improvement in nats (needs ≥ threshold); [95% interval]; A needs the lower bound > 0, B is point estimate only.

| Variant | A (all): Cov Δ | A: NLL impr. | B (M_sep): Cov Δ | B: NLL impr. | C | Verdict |
|---|---|---|---|---|---|---|
| Flow (pre-registered method) | +0.115 [+0.082, +0.148] (pass; vs B4, 0.8785 vs 0.8628) | +0.0954 [+0.0759, +0.1148] (pass; thr. 0.0108; vs B3/cf) | +0.084 [-0.025, +0.187] (**fail**; vs B4, 0.8073 vs 0.7895) | +0.0843 [+0.0084, +0.1617] (pass; thr. 0.0598; vs OFFICIAL/raw) | pass | **FAIL** |
| flowET (added after the FAIL; selected with test data) | +0.158 [+0.127, +0.191] (pass; vs B4, 0.8844 vs 0.8628) | +0.1239 [+0.1040, +0.1445] (pass; thr. 0.0108; vs B3/cf) | +0.127 [+0.022, +0.230] (pass; vs B4, 0.8163 vs 0.7895) | +0.1336 [+0.0577, +0.2086] (pass; thr. 0.0598; vs OFFICIAL/raw) | pass | **PASS** |

**Gate outcome, eye-centred head box (reported alongside).** Δ = relative reduction of the miss rate (needs ≥ 0.10) / NLL improvement in nats (needs ≥ threshold); [95% interval]; A needs the lower bound > 0, B is point estimate only.

| Variant | A (all): Cov Δ | A: NLL impr. | B (M_sep): Cov Δ | B: NLL impr. | C | Verdict |
|---|---|---|---|---|---|---|
| Flow (pre-registered method) | +0.131 [+0.099, +0.162] (pass; vs B4, 0.8816 vs 0.8637) | +0.1032 [+0.0835, +0.1225] (pass; thr. 0.0100; vs B3/cf) | +0.075 [-0.036, +0.181] (**fail**; vs B4, 0.8120 vs 0.7967) | +0.1289 [+0.0561, +0.2055] (pass; thr. 0.0591; vs OFFICIAL/raw) | pass | **FAIL** |
| flowET (added after the FAIL; selected with test data) | +0.172 [+0.142, +0.203] (pass; vs B4, 0.8871 vs 0.8637) | +0.1387 [+0.1182, +0.1594] (pass; thr. 0.0100; vs B3/cf) | +0.111 [+0.002, +0.214] (pass; vs B4, 0.8192 vs 0.7967) | +0.1740 [+0.1004, +0.2495] (pass; thr. 0.0591; vs OFFICIAL/raw) | pass | **PASS** |

> The flowET rows deviate from the pre-registered procedure (§5.2): the variant was added after the first outcome and selected using test data.

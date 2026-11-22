from std.math import round


def fmt_float(val: Float64, decimals: Int) -> String:
    """Format a Float64 with exact fixed decimal places (including trailing zeros)."""
    var factor: Float64 = 1.0
    for _ in range(decimals):
        factor *= 10.0
    var scaled = Int(round(val * factor))
    var neg = scaled < 0
    if neg:
        scaled = -scaled
    var int_div = 1
    for _ in range(decimals):
        int_div *= 10
    var whole = scaled // int_div
    var frac = scaled % int_div
    var out = String("-" if neg else "") + String(whole)
    if decimals == 0:
        return out
    var frac_str = String(frac)
    while frac_str.byte_length() < decimals:
        frac_str = "0" + frac_str
    return out + "." + frac_str


def csv_cell(s: String) -> String:
    """RFC-4180 quote a CSV field if it contains commas or quotes."""
    if "," in s or "\"" in s or "\n" in s:
        var esc = s.replace("\"", "\"\"")
        return "\"" + esc + "\""
    return s


def fmt_sci2(val: Float64) -> String:
    """Format Float64 in %.2e scientific notation (e.g. 2.22e-16)."""
    if val == 0.0:
        return "0.00e+00"
    var neg = val < 0.0
    var x = -val if neg else val
    var exp = 0
    while x >= 10.0 and exp < 308:
        x /= 10.0
        exp += 1
    while x < 1.0 and exp > -308:
        x *= 10.0
        exp -= 1
    var mant = Int(round(x * 100.0))
    if mant >= 1000:
        mant = 100
        exp += 1
    var m_whole = mant // 100
    var m_frac = mant % 100
    var frac_s = ("0" + String(m_frac)) if m_frac < 10 else String(m_frac)
    var exp_sign = "+" if exp >= 0 else "-"
    var exp_abs = -exp if exp < 0 else exp
    var exp_s = ("0" + String(exp_abs)) if exp_abs < 10 else String(exp_abs)
    return String("-" if neg else "") + String(m_whole) + "." + frac_s + "e" + exp_sign + exp_s

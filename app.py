import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import graphviz
from functools import lru_cache

st.set_page_config(page_title="MLM Binary Simulator", layout="wide")
st.title("MLM Binary Network Simulator")

# --- Input Section ---
st.sidebar.header("Input Member Details")
belanja = st.sidebar.number_input("Belanja (Rp)", min_value=0, step=100000, value=2000000)
level_simulasi = st.sidebar.slider("Simulasi Pertumbuhan (Level)", 1, 22, 6)

# --- Bonus Settings ---
st.sidebar.header("\U0001F4B8 Setting Alokasi Bonus")
alokasi_belanja = st.sidebar.number_input("Alokasi dari Belanja (Rp)", min_value=0, step=100000, value=1000000)
bonus_green = st.sidebar.number_input("Bonus GREEN", min_value=0, step=100000, value=5000000)
bonus_silver = st.sidebar.number_input("Bonus SILVER", min_value=0, step=100000, value=10000000)
bonus_red = st.sidebar.number_input("Bonus RED", min_value=0, step=100000, value=50000000)

# --- Helper Functions ---
def build_binary_tree(levels):
    tree = {}
    index = 0
    for level in range(levels + 1):
        nodes = 2 ** level
        tree[level] = list(range(index, index + nodes))
        index += nodes
    return tree

def get_children(index):
    return 2 * index + 1, 2 * index + 2

@lru_cache(maxsize=None)
def count_descendants(member, max_index):
    descendants = set()
    stack = [member]
    while stack:
        node = stack.pop()
        left, right = get_children(node)
        if left <= max_index:
            descendants.add(left)
            stack.append(left)
        if right <= max_index:
            descendants.add(right)
            stack.append(right)
    return descendants

@st.cache_data(show_spinner=False)
def calculate_statuses(all_members, max_index):
    green_members = []
    silver_members = []
    red_members = []

    for member in all_members:
        desc = count_descendants(member, max_index)
        if len(desc) == 14:
            green_members.append(member)

    for member in all_members:
        desc = count_descendants(member, max_index)
        green_count = len([d for d in green_members if d in desc])
        if green_count >= 14:
            silver_members.append(member)

    for member in all_members:
        desc = count_descendants(member, max_index)
        silver_count = len([d for d in silver_members if d in desc])
        if silver_count >= 14:
            red_members.append(member)

    return green_members, silver_members, red_members

def get_status(member, green_members, silver_members, red_members):
    if member in red_members:
        return "Red"
    elif member in silver_members:
        return "Silver"
    elif member in green_members:
        return "Green"
    return "-"

def count_type_descendants(member, max_index, target_members):
    desc = count_descendants(member, max_index)
    return len([d for d in desc if d in target_members])

def format_rupiah(amount):
    return f"Rp{amount:,.0f}".replace(",", ".")

# --- Simulate Tree ---
tree_dict = build_binary_tree(level_simulasi)
all_members = [m for level in tree_dict.values() for m in level]
max_index = max(all_members)

green_members, silver_members, red_members = calculate_statuses(all_members, max_index)

bonus_green_total = len(green_members) * bonus_green
bonus_silver_total = len(silver_members) * bonus_silver
bonus_red_total = len(red_members) * bonus_red

jumlah_member = len(all_members)
total_belanja = jumlah_member * belanja
cash_in = jumlah_member * alokasi_belanja
cash_out = bonus_green_total + bonus_silver_total + bonus_red_total
nett_cash = cash_in - cash_out

# --- Alert Warning ---
if nett_cash < 0:
    st.error(f"\U000026A0\ufe0f Cash In lebih kecil dari Cash Out! Simulasi menyebabkan kerugian sebesar {format_rupiah(-nett_cash)}")

# --- Smart Suggestion ---
max_bonus_green_safe = cash_in // len(green_members) if green_members else 0
min_allocation_required = cash_out // jumlah_member if jumlah_member else 0

st.warning(f"\U0001F4A1 Bonus Green maksimum agar tidak rugi: {format_rupiah(max_bonus_green_safe)}")
st.warning(f"\U0001F4A1 Alokasi minimum yang diperlukan agar tidak rugi: {format_rupiah(min_allocation_required)}")

# --- Output Section ---
st.subheader("\U0001F4CA Ringkasan Simulasi")
st.markdown(f"**Total Member:** {jumlah_member}")
st.markdown(f"**Status Member 0:** {get_status(0, green_members, silver_members, red_members)}")
st.markdown(f"**Bonus Member 0:** {format_rupiah(bonus_green) if 0 in green_members else format_rupiah(bonus_silver) if 0 in silver_members else format_rupiah(bonus_red) if 0 in red_members else 'Rp0'}")

# --- Tabel Alokasi Bonus + Cashflow Lengkap ---
st.subheader("\U0001F4B0 Simulasi Cashflow dan Bonus Alokasi")

data_bonus = {
    "Deskripsi": [
        "Jumlah Member",
        "Belanja per Member",
        "Total Belanja (Rp)",
        "Alokasi untuk Bonus per Member",
        "Total Cash In (Rp)",
        "Jumlah Member Green",
        "Jumlah Member Silver",
        "Jumlah Member Red",
        "Bonus per Green (Rp)",
        "Bonus per Silver (Rp)",
        "Bonus per Red (Rp)",
        "Total Bonus Green (Rp)",
        "Total Bonus Silver (Rp)",
        "Total Bonus Red (Rp)",
        "Total Cash Out (Bonus) (Rp)",
        "Nett (Cash In - Out) (Rp)"
    ],
    "Nilai": [
        f"{jumlah_member:,}",
        format_rupiah(belanja),
        format_rupiah(total_belanja),
        format_rupiah(alokasi_belanja),
        format_rupiah(cash_in),
        f"{len(green_members):,}",
        f"{len(silver_members):,}",
        f"{len(red_members):,}",
        format_rupiah(bonus_green),
        format_rupiah(bonus_silver),
        format_rupiah(bonus_red),
        format_rupiah(bonus_green_total),
        format_rupiah(bonus_silver_total),
        format_rupiah(bonus_red_total),
        format_rupiah(cash_out),
        format_rupiah(nett_cash)
    ]
}

df_bonus = pd.DataFrame(data_bonus)
st.dataframe(df_bonus, use_container_width=True)

# --- Grafik Pertumbuhan ---
st.subheader("\U0001F4C8 Grafik Pertumbuhan Jaringan")
fig, ax = plt.subplots()
level_keys = list(tree_dict.keys())
members_per_level = [len(tree_dict[k]) for k in level_keys]
ax.plot(level_keys, members_per_level, marker='o')
ax.set_xlabel("Level")
ax.set_ylabel("Jumlah Member")
ax.set_title("Pertumbuhan Jaringan Binary")
ax.grid(True)
st.pyplot(fig)

# --- Struktur Binary ---
st.subheader("\U0001F333 Struktur Jaringan Binary")
def draw_binary(start, max_depth):
    dot = graphviz.Digraph()
    queue = [(start, 0)]
    while queue:
        node, level = queue.pop(0)
        if level > max_depth:
            continue
        label = f"#{node}"
        if node in red_members:
            label = f"🔴 {label}"
        elif node in silver_members:
            label = f"⚪ {label}"
        elif node in green_members:
            label = f"🟢 {label}"
        dot.node(str(node), label)
        left, right = get_children(node)
        if left <= max_index:
            dot.edge(str(node), str(left))
            queue.append((left, level + 1))
        if right <= max_index:
            dot.edge(str(node), str(right))
            queue.append((right, level + 1))
    return dot

st.graphviz_chart(draw_binary(0, level_simulasi if level_simulasi <= 6 else 4))

# --- Subtree ---
st.subheader("\U0001F50D Lihat Subjaringan dari Member Tertentu")
selected_node = st.number_input("Masukkan nomor member:", min_value=0, max_value=max_index, step=1)
st.markdown(f"**Status:** {get_status(selected_node, green_members, silver_members, red_members)}")
st.markdown(f"**Bonus:** {format_rupiah(bonus_green) if selected_node in green_members else format_rupiah(bonus_silver) if selected_node in silver_members else format_rupiah(bonus_red) if selected_node in red_members else 'Rp0'}")
st.markdown(f"**Green Downlines:** {count_type_descendants(selected_node, max_index, green_members)}")
st.markdown(f"**Silver Downlines:** {count_type_descendants(selected_node, max_index, silver_members)}")
st.graphviz_chart(draw_binary(selected_node, 3))

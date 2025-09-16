import streamlit as st
from ra_engine import run, RelAlgError, parse_relations, parse_query

st.set_page_config(page_title="Mini-Relax: Relational Algebra Runner", page_icon="🧮", layout="wide")

DEFAULT_REL = """Employees (EID, Name, Age) = {
  "E1", "John", 32
  "E2", "Alice", 28
  "E3", "Bob", 29
}

Departments (DID, DeptName, EID) = {
  "D1", "Sales", "E1"
  "D2", "Engineering", "E2"
  "D3", "HR", "E9"
}"""

DEFAULT_QUERY = 'Employees ⋈_{EID = EID} Departments'

if "relations" not in st.session_state:
    st.session_state.relations = DEFAULT_REL
if "query" not in st.session_state:
    st.session_state.query = DEFAULT_QUERY

st.title("🧮 Mini-Relax — Relational Algebra Runner")
st.write(
    "All-in-one page. The dockbar **only inserts text**; you can freely edit the query. "
    "Engine is from-scratch (no pandas): supports σ (select), π (project), ⋈ (join), ⋃, ∩, −."
)

# Inputs
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Relations input")
    st.text_area(
        "Define one or more relations",
        value=st.session_state.relations,
        height=300,
        help="Format: Name (A, B, C) = {\n  v1, v2, v3\n  ...\n}",
        key="relations",
    )

with col2:
    st.subheader("Query")
    st.caption("Dockbar: click to insert tokens (appends to the end).")
    row1 = ['σ', 'π', '⋈', '⋃', '∩', '−']
    row2 = ['_{', '}', '(', ')', ',', '=']
    row3 = ['>=', '<=', '!=', '"', "'"]

    def insert(tok: str):
        st.session_state.query = (st.session_state.get("query") or "") + tok

    c = st.columns(len(row1))
    for i, t in enumerate(row1):
        if c[i].button(t, use_container_width=True): insert(t)
    c = st.columns(len(row2))
    for i, t in enumerate(row2):
        if c[i].button(t, use_container_width=True): insert(t)
    c = st.columns(len(row3))
    for i, t in enumerate(row3):
        if c[i].button(t, use_container_width=True): insert(t)

    with st.expander("📚 Examples (click to expand/collapse)"):
        st.code('select Age > 30 (Employees)', language="text")
        st.code('π Name (σ Age > 28 (Employees))', language="text")
        st.code('Employees ⋈_{EID = EID} Departments', language="text")
        st.code('join(Employees, Departments, on="EID = EID")', language="text")
        st.code('(π EID,Name (Employees)) ⋃ (π EID,Name (Employees))', language="text")
        st.code('(π EID,Name (Employees)) − (π EID,Name (σ Age > 30 (Employees)))', language="text")
        st.code('π Name,DeptName ( σ Age >= 28 ( Employees ⋈_{EID = EID} Departments ) )', language="text")

    st.text_area(
        "Enter a relational algebra query",
        value=st.session_state.query,
        height=220,
        help="Use symbols (σ, π, ⋈, ⋃, ∩, −) or keywords select/project/join(...).",
        key="query",
    )

# Visualize input relations
st.subheader("👀 Visualize Input Relations")
try:
    env_preview = parse_relations(st.session_state.relations)
    if env_preview:
        for name, rel in env_preview.items():
            st.markdown(f"**{name}** — schema: {rel.schema}  \n_rows: {len(rel.rows)}_")
            st.table([dict(zip(rel.schema, row)) for row in rel.rows] or [])
    else:
        st.info("No relations found.")
except RelAlgError as e:
    st.error(f"{type(e).__name__} while parsing relations: {e}")

# Run
if st.button("▶️ Run", type="primary"):
    try:
        result = run(st.session_state.relations, st.session_state.query)
        st.success("Query executed successfully!")

        tabs = st.tabs(["Result Table", "Result Text", "Parse Details"])
        with tabs[0]:
            if result.rows:
                st.table([dict(zip(result.schema, row)) for row in result.rows])
                csv = result.to_csv()
                st.download_button("Download CSV", data=csv, file_name="result.csv", mime="text/csv")
            else:
                st.info("Empty result set.")
        with tabs[1]:
            st.code(result.pretty(), language="text")
        with tabs[2]:
            try:
                st.markdown("**Original query**")
                st.code(st.session_state.query, language="text")
                ast = parse_query(st.session_state.query)
                st.markdown("**Canonical query**")
                st.code(str(ast), language="text")
                st.markdown("**Relations**")
                for name, rel in env_preview.items():
                    st.markdown(f"- `{name}`: schema = {rel.schema}, rows = {len(rel.rows)}")
            except RelAlgError as e:
                st.error(f"{type(e).__name__}: {e}")

    except RelAlgError as e:
        st.error(f"{type(e).__name__}: {e}")
    except Exception as e:
        st.exception(e)
else:
    st.info("Enter input, build your query with the dockbar, then click **Run**.")
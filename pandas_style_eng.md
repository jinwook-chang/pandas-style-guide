Below is an **English version** of the **pandas Coding Style Guide**, including recommendations on preventing unintended side effects when modifying DataFrame objects. You can adapt and extend this guide to match your team’s specific needs and environment.

---

# pandas Coding Style Guide

## 1. Imports

1. **Basic Imports**  
   ```python
   import pandas as pd
   import numpy as np
   ```
   - Use `import pandas as pd` as the standard import.
   - If you frequently use NumPy, also import `numpy as np`.

2. **Module Alias**  
   - Avoid using aliases other than `pd` for pandas, as it can cause confusion.
   - Unless there is a special circumstance (like name conflicts), stick to `pd`.

---

## 2. Code Layout

1. **Line Length**  
   - Keep line lengths to around 80–100 characters.
   - If a line gets too long, use [implicit line continuation][pep8-line] or the backslash (`\`) to break it into multiple lines.

2. **Line Breaks in Method Chaining**  
   ```python
   # Recommended
   df = (
       pd.read_csv("data.csv")
       .query("value > 0")
       .drop_duplicates()
       .reset_index(drop=True)
   )

   # Not recommended (too many chained methods in one line)
   df = pd.read_csv("data.csv").query("value > 0").drop_duplicates().reset_index(drop=True)
   ```
   - When using methods like `.query()`, `.drop()`, `.groupby()`, `.agg()`, `.merge()`, break lines within parentheses.
   - Easier to read and maintain if you add/remove steps.

3. **Inline Comments**  
   - Prefer writing a single operation per line.  
   - If you need comments, put them on a separate line to clarify the logic.
   ```python
   # Recommended
   # Filter out rows where 'value' is 0 or less, then reset index
   df = (
       df[df["value"] > 0]
       .reset_index(drop=True)
   )

   # Not recommended
   df = df[df["value"] > 0]  # filter out zero or negative values
   df.reset_index(drop=True, inplace=True)
   ```

[pep8-line]: https://peps.python.org/pep-0008/#maximum-line-length "PEP 8: Maximum Line Length"

---

## 3. Variable & Column Naming

1. **Variable Names**  
   - Use snake_case for variable names (e.g., `sales_df`).
   - Avoid abbreviations unless they are widely understood.

2. **DataFrame Column Names**  
   - Use a consistent naming convention, such as snake_case (`sales_amount`) or CamelCase (`SalesAmount`).
   - Minimize temporary columns (like `temp`) unless absolutely necessary.
   - If you must use non-English columns (e.g., in Korean), keep them short and add comments explaining their meaning in English if needed.

3. **Intermediate DataFrame Naming**  
   - For example: `df_raw` → `df_filtered` → `df_final`.
   - Choose descriptive names that indicate the transformation stage.

---

## 4. Data Loading & Saving

1. **Reading Data**  
   ```python
   # CSV
   df = pd.read_csv("input.csv")

   # Excel
   df_excel = pd.read_excel("input.xlsx", sheet_name="Sheet1")

   # Parquet
   df_parquet = pd.read_parquet("input.parquet")
   ```
   - Store file names or paths in variables for easier maintenance.
   - When using many parameters, use parentheses and line breaks for readability.

2. **Saving Data**  
   ```python
   df.to_csv("output.csv", index=False)
   df.to_excel("output.xlsx", index=False, sheet_name="Result")
   df.to_parquet("output.parquet", index=False)
   ```
   - Be explicit with parameters (e.g., `index=False`) in `to_*` functions as well.

---

## 5. Data Selection & Filtering

1. **loc / iloc / at / iat**  
   - Use `df.loc[]` for label-based indexing.
   - Use `df.iloc[]` for integer position-based indexing.
   - Use `df.at[]` or `df.iat[]` for fast access to a single value.
   ```python
   value = df.loc[10, "column_name"]
   value = df.iloc[10, 2]
   ```

2. **Boolean Indexing**  
   - Store filter conditions in variables to improve readability.
   ```python
   # Recommended
   mask_positive = df["value"] > 0
   mask_category = df["category"] == "A"
   df_filtered = df[mask_positive & mask_category]

   # Not recommended
   df_filtered = df[(df["value"] > 0) & (df["category"] == "A")]
   ```

3. **Using `query()`**  
   - Because it uses strings, typos might be hard to catch.
   - If the conditions are simple or your team is comfortable with SQL-like syntax, `query()` can be more readable.
   ```python
   df_filtered = df.query("value > 0 and category == 'A'")
   ```

---

## 6. Data Preprocessing & Transformation

1. **Method Chaining**  
   - Use method chaining to keep transformations in a single flow.
   - If it becomes too long or hard to read, use intermediate variables.

2. **apply, map, applymap**  
   - Prefer **vectorized operations** (e.g., `df["col"] + 10`) over `apply()` for performance.
   - Use `apply()` only when row-wise operations are genuinely needed.
   ```python
   # Recommended
   df["col"] = df["col"] + 10

   # Not recommended (can be slower)
   df["col"] = df["col"].apply(lambda x: x + 10)
   ```

3. **groupby & agg**  
   - When applying multiple aggregations, use a dictionary or method chaining to enhance readability.
   ```python
   df_agg = (
       df.groupby("category")
       .agg(
           mean_value=("value", "mean"),
           sum_value=("value", "sum")
       )
       .reset_index()
   )
   ```
   - You can define `agg_cols = {"value": ["mean", "sum"]}` in a separate variable for reuse.

4. **merge, join**  
   - Specify key parameters like `how`, `on`, `left_on`, `right_on`.
   - Keep a consistent order, such as always merging onto the reference DataFrame first.
   ```python
   df_merged = pd.merge(
       left=df_left,
       right=df_right,
       how="inner",
       on="id"
   )
   ```

---

## 7. Functions & Classes

### 7.1 Encapsulating Repetitive Logic

- If there is repetitive data cleaning or transformation logic, encapsulate it in a function.
- Clearly decide whether the function **modifies** the original DataFrame or **returns a copy**.

```python
def clean_text_column(series: pd.Series) -> pd.Series:
    """
    Perform text cleaning without modifying the original Series in-place.
    Returns a new Series.
    """
    return (
        series.str.lower()
        .str.replace(r"[^a-z0-9]", "", regex=True)
    )

df["clean_text"] = clean_text_column(df["raw_text"])
```

### 7.2 Avoiding Side Effects

> **Side effect**:  
> When a function inadvertently changes external states or its own inputs (e.g., a DataFrame) in an unpredictable way. This can hinder debugging and collaboration.

1. **Do Not Modify the Original DataFrame by Default**  
   - Create a copy (`df.copy()`) inside the function, perform changes, then return the new DataFrame.
   - This preserves the original `df` state.
   ```python
   def remove_outliers(df: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
       """
       Removes rows where 'value_col' is beyond z_threshold standard deviations.
       Returns a new DataFrame without modifying the original.
       """
       df_copy = df.copy()
       
       mean_val = df_copy["value_col"].mean()
       std_val = df_copy["value_col"].std()

       z_score = (df_copy["value_col"] - mean_val).abs() / std_val
       df_copy = df_copy[z_score < z_threshold]

       return df_copy

   df_outliers_removed = remove_outliers(df_raw)
   ```

2. **Avoid `inplace=True`**  
   - `df.drop(columns=["col"], inplace=True)` makes changes directly in the original `df`, which can be harder to track.
   - Instead, return a new DataFrame:
   ```python
   # Recommended
   df_new = df.drop(columns=["col"])  # original df remains unchanged

   # Not recommended
   df.drop(columns=["col"], inplace=True)  # original df is modified in-place
   ```

3. **If You Must Modify the Original DataFrame**  
   - In rare cases (e.g., memory constraints with large datasets), in-place modification is necessary.
   - **Clearly label** the function or method as in-place in its name and/or docstring.
   ```python
   def normalize_inplace(df: pd.DataFrame, col: str) -> None:
       """
       Normalizes the specified column to a 0–1 range.
       Modifies the original DataFrame in-place (no return value).
       """
       min_val = df[col].min()
       max_val = df[col].max()
       df[col] = (df[col] - min_val) / (max_val - min_val)

   normalize_inplace(df_raw, "value_col")  # df_raw is now changed
   ```

4. **Clarity on Function Input/Output**  
   - Clearly document what the function returns and whether it mutates the input DataFrame.
   - As a general rule, “functions do not mutate the input `DataFrame`” is easiest for collaboration, with explicit exceptions.

### 7.3 Class Usage

- If you maintain state in `self.df` within a class, be explicit about whether methods update `self.df`.
- If “original data preservation” is crucial, create a copy in each method, and return the new DataFrame instead of mutating `self.df`.
- If a method updates `self.df`, clearly note it in the method name or docstring (e.g., “updates `self.df` in-place”).

---

## 8. Exception Handling & Logging

1. **Exception Handling**  
   - Use `try-except` blocks where errors are likely (e.g., file reading, type conversions).
   ```python
   try:
       df = pd.read_csv("data.csv")
   except FileNotFoundError as e:
       print(f"File not found: {e}")
       # Additional handling...
   ```

2. **Logging**  
   - Prefer Python’s built-in `logging` module with log levels (`DEBUG`, `INFO`, `WARNING`, `ERROR`) to simple `print()` statements.
   - Essential in production code for easier debugging and monitoring.

---

## 9. Performance Considerations

1. **Vectorization**  
   - Whenever possible, use built-in vectorized operations (e.g., `df["col"] + 10`) over `apply()`.
   - Avoid row-by-row loops (`for`, `while`) and `iterrows()` unless absolutely necessary.

2. **Memory Usage**  
   - For large datasets, specify `dtype` to reduce memory usage.
   ```python
   df = pd.read_csv("big_data.csv", dtype={
       "id": "int32",
       "category": "category",
       "value": "float32"
   })
   ```

3. **Parallel/Distributed Processing**  
   - Consider libraries like `dask` or `modin` for parallelized DataFrame operations.
   - Note that these may differ slightly from pandas APIs, so ensure the team understands them to avoid confusion.

---

## 10. Comprehensive Example

Below is a sample script illustrating the recommended practices:

```python
import pandas as pd
import numpy as np

def remove_outliers(df: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
    """
    Removes rows whose 'value_col' is beyond z_threshold standard deviations.
    Returns a new DataFrame, leaving the original df unmodified.
    """
    df_copy = df.copy()

    mean_val = df_copy["value_col"].mean()
    std_val = df_copy["value_col"].std()
    
    z_score = (df_copy["value_col"] - mean_val).abs() / std_val
    df_copy = df_copy[z_score < z_threshold]
    
    return df_copy

def main_pipeline(input_path: str, output_path: str) -> None:
    """
    Reads a CSV file, applies transformations, and saves the result as a CSV.
    Does not mutate the original DataFrame.
    """
    # Read data
    df_raw = pd.read_csv(input_path)

    # Basic preprocessing: drop unnecessary column, fill missing values
    df_cleaned = (
        df_raw
        .drop(columns=["unused_col"])   # returns a new DF, original df_raw is unchanged
        .fillna({"value_col": 0})       # returns a new DF
    )
    df_cleaned = remove_outliers(df_cleaned, z_threshold=3.0)

    # Group & aggregate
    df_agg = (
        df_cleaned
        .groupby("category_col")
        .agg(
            mean_value=("value_col", "mean"),
            count=("value_col", "count")
        )
        .reset_index()
    )

    # Save result
    df_agg.to_csv(output_path, index=False)

if __name__ == "__main__":
    input_file = "data/input.csv"
    output_file = "data/output.csv"

    main_pipeline(input_file, output_file)
```

---

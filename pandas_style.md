아래는 **pandas**를 활용할 때 추천하는 코딩 스타일 가이드 전체를 **Markdown** 형식으로 정리한 문서입니다.  
코드 예시와 함께, 원본 `DataFrame`을 의도치 않게 변경하는 **사이드 이펙트(side effect)**를 방지하기 위한 팁도 추가되어 있습니다.  
조직/프로젝트 상황에 맞게 자유롭게 수정해 사용하시면 됩니다.

---

# pandas 코딩 스타일 가이드

## 1. 임포트 방식

1. **기본 임포트**  
   ```python
   import pandas as pd
   import numpy as np
   ```
   - `import pandas as pd`를 표준으로 합니다.
   - NumPy 연산을 자주 사용하는 경우 `import numpy as np`를 기본으로 합니다.

2. **모듈 Alias 사용**  
   - `pd` 외 다른 alias를 사용하는 것은 혼동을 줄 수 있으므로 지양합니다.
   - 프로젝트 특수 상황(동일 이름 모듈 충돌 등)이 아니라면 `pd`만 고정합니다.

---

## 2. 코드 레이아웃

1. **줄 길이 제한**  
   - 한 줄에 80~100자를 넘지 않도록 합니다.
   - 줄이 길어지면, 파이썬의 [implicit line continuation][pep8-line] 또는 `\`를 사용하여 줄을 나눕니다.

2. **함수/메서드 체이닝 시 줄바꿈**  
   ```python
   # 권장
   df = (
       pd.read_csv("data.csv")
       .query("value > 0")
       .drop_duplicates()
       .reset_index(drop=True)
   )

   # 비권장(한 줄에 너무 많은 체이닝)
   df = pd.read_csv("data.csv").query("value > 0").drop_duplicates().reset_index(drop=True)
   ```
   - `.query()`, `.drop()`, `.groupby()`, `.agg()`, `.merge()` 등 메서드 체이닝 시 괄호 안에서 줄바꿈.
   - 체이닝 단계가 추가/삭제될 때 수정이 용이해집니다.

3. **인라인 주석**  
   - 한 줄에 여러 가지 로직을 담지 않고, 한 줄에 한 기능만 작성합니다.
   - 주석이 필요하다면 가급적 별도 줄에 달아서 의미를 명확히 표현합니다.
   ```python
   # 권장
   # 0 이하인 값을 제거하고, 인덱스를 초기화한다.
   df = (
       df[df["value"] > 0]
       .reset_index(drop=True)
   )

   # 비권장
   df = df[df["value"] > 0]  # 0 이하 값 제거 후 인덱스 초기화
   df.reset_index(drop=True, inplace=True)
   ```

[pep8-line]: https://peps.python.org/pep-0008/#maximum-line-length "파이썬 PEP 8의 줄 길이 가이드"

---

## 3. 변수 및 컬럼명

1. **변수명**  
   - 스네이크 케이스(snake_case)를 사용합니다.
   - 약어 대신 풀어서 의미를 명확히 표현합니다(예: `sales_df`).

2. **데이터프레임 컬럼명**  
   - 컬럼명도 스네이크 케이스 혹은 일관된 스타일(예: `SalesAmount`, `sales_amount`)을 사용합니다.
   - 로직 상 필요한 경우 외에는 임시 컬럼(`temp` 등)을 최소화합니다.
   - 한글 컬럼명이 필요한 경우, 직관적으로 짧게 작성하고 필요 시 주석으로 영문 설명을 덧붙입니다.

3. **중간 결과 데이터프레임 명명**  
   - 예시: `df_raw` → `df_filtered` → `df_final`.
   - 수행 동작을 반영하는 이름을 사용해, 데이터가 어떤 단계인지 명확히 알 수 있도록 합니다.

---

## 4. 데이터 불러오기/저장

1. **데이터 읽기 예시**  
   ```python
   # CSV
   df = pd.read_csv("input.csv")

   # Excel
   df_excel = pd.read_excel("input.xlsx", sheet_name="Sheet1")

   # Parquet
   df_parquet = pd.read_parquet("input.parquet")
   ```
   - 파일명이나 경로명을 변수에 저장해두면 유지보수에 좋습니다.
   - 파라미터가 많은 경우 괄호와 줄바꿈을 통해 가독성을 높입니다.

2. **데이터 저장 예시**  
   ```python
   df.to_csv("output.csv", index=False)
   df.to_excel("output.xlsx", index=False, sheet_name="Result")
   df.to_parquet("output.parquet", index=False)
   ```
   - `to_*` 계열 함수에서도 `index=False` 등 파라미터를 명시적으로 작성합니다.

---

## 5. 데이터 선택/필터링

1. **loc / iloc / at / iat**  
   - **레이블** 기반 선택: `df.loc[]`
   - **정수 위치** 기반 선택: `df.iloc[]`
   - **단일 값**에 빠르게 접근: `df.at[]`, `df.iat[]`  
   ```python
   value = df.loc[10, "column_name"]
   value = df.iloc[10, 2]
   ```

2. **Boolean Indexing**  
   - 필터 조건을 변수로 분리하여 가독성을 높입니다.
   ```python
   # 권장
   mask_positive = df["value"] > 0
   mask_category = df["category"] == "A"
   df_filtered = df[mask_positive & mask_category]

   # 비권장
   df_filtered = df[(df["value"] > 0) & (df["category"] == "A")]
   ```

3. **query() 사용 여부**  
   - 문자열로 작성하기 때문에 오타 등 실수 발견이 어려울 수 있습니다.
   - 간단한 조건이거나 SQL 문법에 익숙한 팀원과 작업할 때는 `query()`가 직관적일 수 있습니다.
   ```python
   df_filtered = df.query("value > 0 and category == 'A'")
   ```

---

## 6. 데이터 전처리/변환

1. **체이닝 사용**  
   - 가능한 한 메서드 체이닝을 사용하여 한 흐름 안에서 연산 과정을 나열합니다.
   - 가독성이 떨어지면 중간 변수를 적절히 사용합니다.

2. **apply, map, applymap**  
   - 간단한 변환은 **벡터화 연산**(`df["col"] + 10` 등)을 우선 사용합니다.
   - 불가피하게 **row 단위 처리**가 필요한 경우 `apply()`를 사용합니다.
   ```python
   # 권장
   df["col"] = df["col"] + 10

   # 비권장(성능 저하)
   df["col"] = df["col"].apply(lambda x: x + 10)
   ```

3. **groupby, agg**  
   - 여러 개의 집계 함수 적용 시 딕셔너리 또는 체이닝을 사용해 가독성을 높입니다.
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
   - 별도의 `agg_cols = {"value": ["mean", "sum"]}`로 정의해두고 사용하면 재사용하기 편합니다.

4. **merge, join**  
   - 병합 시 `how`, `on`(또는 `left_on`, `right_on`) 등 핵심 파라미터를 명시합니다.
   - 레퍼런스가 되는 DF를 앞에, 붙일 DF를 뒤에 두는 방식을 일관성 있게 유지합니다.
   ```python
   df_merged = pd.merge(
       left=df_left,
       right=df_right,
       how="inner",
       on="id"
   )
   ```

---

## 7. 함수/클래스 활용

### 7.1 반복되는 로직의 함수화

- 특정 컬럼들에 대해 반복 적용하는 전처리 로직이 있다면, 함수로 만들어 모듈화합니다.
- 함수가 **원본 DataFrame을 변경**하는지, **새로운 DataFrame을 반환**하는지 합의해 둡니다.

```python
def clean_text_column(series: pd.Series) -> pd.Series:
    """텍스트 전처리를 수행하는 함수 - 원본 Series를 변경하지 않고 새 Series 반환"""
    return (
        series.str.lower()
        .str.replace(r"[^a-z0-9]", "", regex=True)
    )

df["clean_text"] = clean_text_column(df["raw_text"])
```

### 7.2 사이드이펙트 방지(원본 DataFrame 무변경)

> **사이드 이펙트(side effect)**:  
> 함수가 외부 상태나 함수 인자(예: `df`)의 상태를 **의도치 않게** 변경하여 예측 가능성을 떨어뜨리는 것.  
> 팀원 간 협업 시 코드 이해도와 디버깅을 어렵게 만들므로 주의가 필요합니다.

1. **원본 DataFrame을 변경하지 않는 방식을 기본으로**  
   - 함수 내부에서 `df.copy()`를 사용해 복사본을 만든 뒤 이를 변경하고 반환합니다.
   - 함수 호출 전후로 원본 상태가 유지되어 예측 가능성이 높아집니다.
   ```python
   def remove_outliers(df: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
       """
       표준편차 z_threshold 배 이상의 값을 제거한 새로운 DataFrame을 반환.
       원본 df는 변경되지 않음.
       """
       df_copy = df.copy()
       
       mean_val = df_copy["value_col"].mean()
       std_val = df_copy["value_col"].std()

       z_score = (df_copy["value_col"] - mean_val).abs() / std_val
       df_copy = df_copy[z_score < z_threshold]

       return df_copy

   # 사용 예시
   df_outliers_removed = remove_outliers(df_raw, z_threshold=3.0)
   ```

2. **`inplace=True` 사용 지양**  
   - `df.drop(columns=["col"], inplace=True)`와 같이 원본을 직접 바꾸는 방식은 추적이 까다롭습니다.
   - 새 DataFrame을 반환받아 사용하는 방식을 기본으로 합니다.
   ```python
   # 권장
   df_new = df.drop(columns=["col"])  # 원본 df는 변경되지 않음

   # 비권장
   df.drop(columns=["col"], inplace=True)  # df가 즉시 변경됨
   ```

3. **부득이하게 원본을 변경해야 하는 경우**  
   - 메모리 절약, 성능 최적화 등 이유로 원본 변경이 필요한 경우 함수 이름, docstring 등에 **원본이 변경**됨을 명시합니다.
   ```python
   def normalize_inplace(df: pd.DataFrame, col: str) -> None:
       """
       주어진 col 값을 정규화 (0~1 스케일)하며,
       원본 DataFrame을 직접 변경(in-place)하는 함수.
       """
       min_val = df[col].min()
       max_val = df[col].max()
       df[col] = (df[col] - min_val) / (max_val - min_val)
       # 반환값 없음, df가 직접 변경됨

   # 사용 예시
   normalize_inplace(df_raw, "value_col")  # df_raw 직접 변경
   ```

4. **함수의 입력/출력 명시성 유지**  
   - 함수가 반환하는 결과와 원본 변경 여부를 코드에서 바로 파악할 수 있도록 주석, 함수명, docstring 등을 활용합니다.
   - 팀 내에서 “함수 호출 시 원본이 바뀌지 않는다”를 원칙으로 하고, 예외가 필요한 경우에만 원본 변경 함수를 씁니다.

### 7.3 클래스 활용 시 주의사항

- 클래스 내부에서 `self.df` 같은 멤버 변수를 직접 변경하는 경우, 각 메서드가 `self.df`를 업데이트하는지 여부를 명시하십시오.
- “원본 유지”가 중요하면 메서드에서 `self.df`의 복사본을 만들고 반환하도록 합니다.
- “멤버 변수를 업데이트”하는 로직이라면, 메서드명 혹은 docstring에 “**self.df가 갱신됨**”을 분명히 알립니다.

---

## 8. 예외 처리 & 로깅

1. **예외 처리**  
   - 파일 불러오기, 형 변환 등 오류 발생 가능성이 큰 부분에서는 `try-except` 구문으로 흐름을 제어합니다.
   ```python
   try:
       df = pd.read_csv("data.csv")
   except FileNotFoundError as e:
       print(f"파일을 찾을 수 없습니다: {e}")
       # 추가 예외 처리 로직
   ```

2. **로깅**  
   - 단순 `print()` 대신 Python의 `logging` 모듈을 사용해 로그 레벨(`DEBUG`, `INFO`, `WARNING`, `ERROR`)을 설정합니다.
   - 팀 전체 공통 코드에서는 필수적으로 적용해, 문제가 발생했을 때 추적이 쉽도록 합니다.

---

## 9. 성능 고려 사항

1. **벡터화**  
   - 가능하면 `apply()` 대신 벡터화 연산(예: `df["col"] + 10`) 및 내장 함수(`.str`, `.fillna()`, `.replace()`) 등을 우선합니다.
   - 루프(`for`, `while`)나 `iterrows()` 같은 row 단위 접근은 피합니다.

2. **메모리 사용**  
   - 대용량 데이터의 경우, `dtype`을 명시하여 메모리 사용량을 줄일 수 있습니다.
   ```python
   df = pd.read_csv("big_data.csv", dtype={
       "id": "int32",
       "category": "category",
       "value": "float32"
   })
   ```

3. **병렬 처리**  
   - `dask`, `modin` 등 병렬 처리를 지원하는 라이브러리를 검토할 수 있습니다.
   - pandas API와 조금 달라질 수 있으므로, 팀원들이 모두 숙지하고 있어야 혼란이 없습니다.

---

## 10. 예시 종합

아래 예시 코드는 위의 가이드를 종합적으로 반영한 형태입니다.

```python
import pandas as pd
import numpy as np

def remove_outliers(df: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
    """
    표준편차 z_threshold 배 이상의 값을 제거한 새로운 DataFrame을 반환.
    원본 df는 변경되지 않음.
    """
    df_copy = df.copy()

    mean_val = df_copy["value_col"].mean()
    std_val = df_copy["value_col"].std()
    
    z_score = (df_copy["value_col"] - mean_val).abs() / std_val
    df_copy = df_copy[z_score < z_threshold]
    
    return df_copy

def main_pipeline(input_path: str, output_path: str) -> None:
    """
    CSV 파일을 읽어와 전처리 후, 결과를 CSV로 저장하는 메인 파이프라인 함수.
    원본 DataFrame을 변경하지 않는 방식을 사용한다.
    """
    # 데이터 읽기
    df_raw = pd.read_csv(input_path)

    # 전처리: 불필요 컬럼 제거, 결측치 처리, 이상치 제거
    df_cleaned = (
        df_raw
        .drop(columns=["unused_col"])  # 원본은 변경되지 않고, 새 DF 반환
        .fillna({"value_col": 0})      # 원본은 변경되지 않고, 새 DF 반환
    )
    df_cleaned = remove_outliers(df_cleaned, z_threshold=3.0)

    # 집계
    df_agg = (
        df_cleaned
        .groupby("category_col")
        .agg(
            mean_value=("value_col", "mean"),
            count=("value_col", "count")
        )
        .reset_index()
    )

    # 결과 저장
    df_agg.to_csv(output_path, index=False)

if __name__ == "__main__":
    input_file = "data/input.csv"
    output_file = "data/output.csv"

    main_pipeline(input_file, output_file)
```

---

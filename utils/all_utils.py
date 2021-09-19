def prepare_data(df):
  """this method prepare data for input

  Args:
      df (dataframe): input data frame

  Returns:
      tupple: tupple of the data
  """
  X = df.drop("y", axis=1)

  y = df["y"]

  return X, y
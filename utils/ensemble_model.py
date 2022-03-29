from fastai.tabular.all import *

class EnsembleModel:
  def __init__(self, model_component):
    self.to = model_component[0]
    self.model = model_component[1]

  def predict(self, inputs):
    # Do something here with self.models to calculate your predictions.
    # then return them.
    to_new = self.to.train.new(inputs)
    to_new.process()
    xs = to_new.xs
    predictions = self.model.predict(xs)

  
      

    
    return predictions


class PretrainerLoading():
  def __init__(self, filename):

    self.filename = filename

  def saving_preprocessor(self, to):

    return to.export(self.filename)

  def loading_preprocessor(self, df):

    to_load = load_pandas(self.filename)
    to_new = to_load.train.new(df)
    to_new.process()
    return to_new.xs






@patch
def export(self:TabularPandas, fname='export.pkl', pickle_protocol=2):
    "Export the contents of `self` without the items"
    old_to = self
    self = self.new_empty()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pickle.dump(self, open(Path(fname), 'wb'), protocol=pickle_protocol)
        self = old_to

def load_pandas(fname):
    "Load in a `TabularPandas` object from `fname`"
    distrib_barrier()
    res = pickle.load(open(fname, 'rb'))
    return res

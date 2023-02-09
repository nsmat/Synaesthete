# Synaesthete

Synaesthete is a repo for combining audio input with reactive visualisations.

For an example workflow, see test.py.

The code is structured around two classes:
  - __Performance__ this object is responsible for processing an audio input, and using a list of specified effets to visualise it.
  - __Effects__ these objects are user specified, and are responsible for the visualisation. It's only requirements are that a) they must take the data provided by Performance as input, and b) the get_image method should return an iterable of Matplotlib artists.
  
  

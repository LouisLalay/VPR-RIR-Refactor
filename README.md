# This package is a refactored version of VPR-RIR

## Initial repo

The initial version [VPR-RIR](https://github.com/LouisLalay/VPR-RIR) is now outdated and should not be used anymore.

## Run the code yourself

We use `uv` to manage the environment. You can install it with the following command:

```bash
pip install uv
```
and then create the python environment that contains all the dependencies with the following command:

```bash
uv sync --all-extras
```

You can launch the experiments with:

```bash
uv run main.py
```

## References

This work is currently not published. If you use this code, please cite the following papers:

```bibtex
@inproceedings{rodriguesTaylorSWFTFastDiscrete2026,
  title     = {{Taylor-SWFT: fast discrete Statistical Wave Field Theory using Taylor expansion for late reverberation}},
  author    = {Rodrigues, Marius and Lalay, Louis and Badeau, Roland and Richard, Ga{\"e}l and Fontaine, Mathieu},
  booktitle = {Forum Acusticum 2026},
  year      = {2026},
  month     = {Sep},
  keywords  = {reverberation statistical wave field theory dynamic room acoustic simulation physics-based models ; reverberation ; statistical wave field theory ; dynamic room acoustic simulation ; physics-based models}
}
@inproceedings{lalayUnifiedVariationalPhysicsaware2025,
  title      = {Unified {{Variational}} and {{Physics}}-Aware {{Model}} for {{Room Impulse Response Estimation}}},
  author     = {Lalay, Louis and Fontaine, Mathieu and Badeau, Roland},
  date       = {2025},
  pages      = {3818--3822},
  doi        = {10.21437/Interspeech.2025-1942},
  url        = {https://www.isca-archive.org/interspeech_2025/lalay25_interspeech.html},
  urldate    = {2026-10-05},
  eventtitle = {Proc. {{Interspeech}} 2025},
}
```
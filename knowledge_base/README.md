# Personal Assistant Knowledge Base

This directory contains private JSON data used by the local assistant. The files are intentionally separated from the model and application code.

Current files:

- personal_data.json — structured records for financial, medical, document, and professional-history information.

The assistant should only use records that are explicitly available in this directory. It must not invent missing information. Users should review and control which JSON data is placed here.

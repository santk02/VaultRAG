# Document Sources

This document logs the sources of sample documents used in VaultRAG for testing and demonstration purposes.

## Banking Documents

### Sample Documents (Placeholder)

*Note: Replace with actual document sources when adding real documents*

**Sources:**
- RBI notifications (rbi.org.in)
- SEBI circulars (sebi.gov.in) 
- SEC EDGAR filings (sec.gov/edgar)
- BIS Basel documents (bis.org/bcbs)

**Document Types:**
- Compliance policies
- Regulatory guidelines
- Annual reports
- Risk management frameworks

## Healthcare Documents

### Sample Documents (Placeholder)

*Note: Replace with actual document sources when adding real documents*

**Sources:**
- FDA drug labels (accessdata.fda.gov)
- WHO guidelines (who.int/publications)
- CMS manuals (cms.gov)
- PubMed Central open access

**Document Types:**
- Clinical guidelines
- Regulatory requirements
- Patient privacy policies
- Healthcare compliance frameworks

## Document Guidelines

### File Naming Convention

- Use descriptive filenames that reflect content
- Include source abbreviation (e.g., "rbi_circular_2024.pdf")
- Avoid spaces, use underscores
- Keep filenames under 100 characters

### Content Requirements

- Documents should be text-based (not scanned images)
- Include page numbers for accurate citation
- 5-50 pages per document
- Mixed complexity levels (simple to complex)

### Metadata

For each document, log:
- Source URL or origin
- Download date
- Document type
- Page count
- Content summary

## Adding New Documents

1. Download document from official source
2. Verify it contains extractable text
3. Rename according to naming convention
4. Add to appropriate subdirectory (banking/healthcare)
5. Update this sources.md file with metadata
6. Test ingestion pipeline
7. Verify retrieval quality

## Legal Compliance

- All sample documents are from publicly available sources
- Documents are used for testing and demonstration only
- No confidential or proprietary information included
- Respect copyright and usage terms of source documents

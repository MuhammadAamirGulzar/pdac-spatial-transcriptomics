#!/usr/bin/env Rscript
# =============================================================================
# Split the published 30-sample Seurat object into per-sample artefacts.
#
# Source : Zenodo 10.5281/zenodo.10712047  ->  PDAC_Updated.rds  (11.6 GB on disk)
# Paper  : Khaliq et al., Nat Genet 2024 (PMID 39294496) == GEO GSE272362
#
# MEMORY  ---------------------------------------------------------------------
# A single sample measures 898 MB in RAM (Spatial 426 + SCT 231 + integrated 226).
# Scaled to 91,496 spots the full object is ~25 GB decompressed, and readRDS is
# atomic -- there is no partial-read escape hatch.  You need a machine with
# >= 32 GB RAM (48 GB comfortable).  This will NOT run on an 8 GB laptop.
#
# The script drops SCT / integrated / scale.data immediately after load, which
# frees roughly half the footprint before any per-sample work begins.
#
# WHAT IT WRITES  -------------------------------------------------------------
#   <out>/ST/<sample>.rds                 slim Seurat object (Spatial counts +
#                                         rctd_fullfinal + fges + image)
#   <out>/scVI_counts/<sample>.csv        genes x barcodes raw counts
#                                         (same orientation as the existing files)
#   <out>/scVI_counts/<sample>_qc_metrics.csv
#   <out>/rctd/<sample>_rctd_fullfinal.csv   spots x 15 published proportions
#   <out>/fges/<sample>_fges.csv             spots x 27 Bagaev signatures
#   <out>/coords/<sample>_coords.csv         barcode,row,col,imagerow,imagecol
#   <out>/cohort_metadata.csv                one row per sample, clinical fields
#   <out>/patient_map.csv                    sample -> patient (for LOPO folds)
#
# REQUIREMENTS  ---------------------------------------------------------------
#   Seurat, SeuratObject, Matrix, data.table
#
# USAGE  ----------------------------------------------------------------------
#   Rscript split_full_cohort.R <path-to-PDAC_Updated.rds> <output-dir>
# =============================================================================

suppressMessages({
  library(Seurat); library(SeuratObject); library(Matrix); library(data.table)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: Rscript split_full_cohort.R <PDAC_Updated.rds> <outdir>")
rds_path <- args[1]
out_root <- args[2]

# QC thresholds -- must match 02_Gene_Export/scvi-gene-export.ipynb, which is what
# produced the `pass_qc` column in the existing dataset/ST/scVI_counts/*_qc_metrics.csv.
MIN_GENES  <- 200
MIN_COUNTS <- 400

# data.table writes the counts CSVs with every core.  Verified on IU_PDA_T4
# (17,893 x 3,621): write.csv 44.2 s -> fwrite 1.2 s, md5-IDENTICAL output.
setDTthreads(0)

for (d in c("ST", "scVI_counts", "rctd", "fges", "coords")) {
  dir.create(file.path(out_root, d), recursive = TRUE, showWarnings = FALSE)
}

msg <- function(...) cat(sprintf("[%s] ", format(Sys.time(), "%H:%M:%S")), ..., "\n", sep = "")
mem <- function() msg("  RAM in use: ", round(sum(gc()[, 2]) / 1024, 2), " GB")

# Write a genes x barcodes count matrix byte-for-byte the way the original
# 02_Gene_Export/scvi-gene-export.ipynb wrote dataset/ST/scVI_counts/<sample>.csv:
# `write.csv(as.data.frame(as.matrix(counts)), row.names = TRUE)` -- i.e. QUOTED
# gene names and barcodes, and a leading empty header field.  ~35x faster (verified
# on IU_PDA_T4: 44.2 s -> 1.3 s, md5 identical).
#
# Counts are integers; storing them as such halves peak RAM vs the double matrix
# `as.matrix()` returns and removes all float formatting work.
write_counts_csv <- function(m, path) {
  dm <- as.matrix(m)
  storage.mode(dm) <- "integer"
  dt <- cbind(data.table(rownames(m)), as.data.table(dm))
  setnames(dt, c("", colnames(m)))
  fwrite(dt, path, quote = TRUE, nThread = getDTthreads())
  rm(dm, dt); invisible(gc(FALSE))
}

# spots x features matrix (RCTD proportions, fges scores) -> barcode-keyed CSV.
# `assay` and `subdir` differ: the RCTD assay is `rctd_fullfinal`, its output dir `rctd`.
write_spot_matrix <- function(obj_sub, sample, assay, subdir, suffix) {
  x <- t(as.matrix(GetAssayData(obj_sub, assay = assay, layer = "data")))
  dt <- cbind(data.table(barcode = rownames(x)), as.data.table(x))
  fwrite(dt, file.path(out_root, subdir, paste0(sample, suffix)),
         quote = FALSE, nThread = getDTthreads())
  invisible(NULL)
}

# ---------------------------------------------------------------- load
msg("reading ", rds_path, " (", round(file.size(rds_path) / 1024^3, 2), " GB on disk) ...")
msg("  this takes several minutes and needs ~25 GB RAM")
obj <- readRDS(rds_path)
# The stored DefaultAssay is rctd_fullfinal (15 features), so report Spatial's
# gene count rather than nrow(obj), which would print a misleading "15 genes".
DefaultAssay(obj) <- "Spatial"
msg("loaded: ", nrow(obj), " genes x ", ncol(obj), " spots")
msg("  assays    : ", paste(Assays(obj), collapse = ", "))
msg("  images    : ", length(Images(obj)), " -> ", paste(head(Images(obj), 40), collapse = ", "))

# The published object was written by a SeuratObject version whose VisiumV1 class
# had no `misc` slot.  SeuratObject >= 5 declares it, so validObject() rejects each
# image the first time subset() assigns it back -- "slots in class definition but
# not in object: misc".  Backfill the slot on load; it is empty by definition.
n_patched <- 0
for (nm in names(obj@images)) {
  if (!"misc" %in% names(attributes(obj@images[[nm]]))) {
    img <- obj@images[[nm]]
    attr(img, "misc") <- list()
    obj@images[[nm]] <- img
    n_patched <- n_patched + 1
  }
}
if (n_patched) msg("  patched ", n_patched, " image(s) missing the VisiumV1 `misc` slot")
mem()

# ---------------------------------------------------------------- identify samples
id_col <- if ("orig.ident" %in% colnames(obj@meta.data)) "orig.ident" else NULL
if (is.null(id_col)) stop("no orig.ident column -- cannot identify samples")
samples <- sort(unique(as.character(obj@meta.data[[id_col]])))
msg("samples (", length(samples), "): ", paste(samples, collapse = ", "))

if (!"rctd_fullfinal" %in% Assays(obj))
  warning("rctd_fullfinal assay absent -- cell modality will not be exported")

# ---------------------------------------------------------------- shed weight
keep_assays <- intersect(c("Spatial", "rctd_fullfinal", "rctd_full_all", "fges"), Assays(obj))
msg("dropping assays: ", paste(setdiff(Assays(obj), keep_assays), collapse = ", "))
DefaultAssay(obj) <- "Spatial"
obj <- DietSeurat(obj, assays = keep_assays, layers = c("counts", "data"),
                  dimreducs = NULL, graphs = NULL)
gc(); mem()

# ---------------------------------------------------------------- clinical metadata
clin_cols <- intersect(c("orig.ident", "patient", "Origin", "Treatment", "Neoadjuvant_Chemo",
                         "Age", "Ethnic", "Gender", "Location_in_pancreas", "AJCC_stage",
                         "Histology", "Tumor_Grade_Histology", "R0_ressection",
                         "Peripheral_Nerve_Invasion", "Lymphovascular_Invasion",
                         "Nodes_collected", "Nodes_positive", "DV200_.", "SlideName",
                         "Sample_ID2", "SeuratID_OLD", "AreaCode",
                         "Data_of_Surgery_start_point_in_survival_analysis"),
                       colnames(obj@meta.data))

cohort <- do.call(rbind, lapply(samples, function(s) {
  i <- which(obj@meta.data[[id_col]] == s)[1]
  row <- obj@meta.data[i, clin_cols, drop = FALSE]
  row[] <- lapply(row, as.character)
  cbind(sample = s, n_spots = sum(obj@meta.data[[id_col]] == s), row)
}))
rownames(cohort) <- NULL
write.csv(cohort, file.path(out_root, "cohort_metadata.csv"), row.names = FALSE)
msg("wrote cohort_metadata.csv")

if ("patient" %in% colnames(obj@meta.data)) {
  pmap <- unique(data.frame(
    sample  = as.character(obj@meta.data[[id_col]]),
    patient = as.character(obj@meta.data$patient), stringsAsFactors = FALSE))
  pmap <- pmap[order(pmap$patient, pmap$sample), ]
  write.csv(pmap, file.path(out_root, "patient_map.csv"), row.names = FALSE)
  msg("wrote patient_map.csv -- ", nrow(pmap), " samples across ",
      length(unique(pmap$patient)), " patients")
  dup <- pmap$patient[duplicated(pmap$patient)]
  if (length(dup))
    msg("  NOTE matched pairs (same patient, >1 sample): ",
        paste(sort(unique(dup)), collapse = ", "),
        "  -> folds MUST be leave-one-patient-out")
}

# ---------------------------------------------------------------- per-sample export
# Resolve sample -> image ONCE.  The original form rescanned every image's full
# coordinate table for every sample (O(samples x images x spots)); with 30 samples
# and 91k spots that is ~30x more matching work than needed.
msg("mapping samples -> images ...")
barcode_to_image <- new.env(hash = TRUE, parent = emptyenv())
for (im in Images(obj)) {
  for (bc in rownames(obj@images[[im]]@coordinates)) assign(bc, im, envir = barcode_to_image)
}
sample_image <- vapply(samples, function(s) {
  bcs <- colnames(obj)[obj@meta.data[[id_col]] == s]
  hit <- NA_character_
  for (bc in bcs) if (exists(bc, envir = barcode_to_image, inherits = FALSE)) {
    hit <- get(bc, envir = barcode_to_image); break
  }
  hit
}, character(1))
img_for <- function(s) unname(sample_image[[s]])
msg("  matched ", sum(!is.na(sample_image)), "/", length(samples), " samples to an image")

summary_rows <- list()
for (s in samples) {
  msg("=== ", s, " ===")
  cells <- colnames(obj)[obj@meta.data[[id_col]] == s]
  sub <- subset(obj, cells = cells)

  # ---- counts: genes x barcodes, matching the existing scVI_counts orientation
  cm <- GetAssayData(sub, assay = "Spatial", layer = "counts")

  # QC metrics are reported for EVERY spot (pre-filter), exactly as the original
  # scvi-gene-export notebook did.  It read nFeature_Spatial / nCount_Spatial from
  # the metadata; recomputing from the count matrix reproduces both exactly.
  nGene <- Matrix::colSums(cm > 0)
  nUMI  <- Matrix::colSums(cm)
  qcdf  <- data.frame(barcode = colnames(cm), nFeature = as.integer(nGene),
                      nCount = as.numeric(nUMI))
  qcdf$pass_qc <- (qcdf$nFeature >= MIN_GENES) & (qcdf$nCount >= MIN_COUNTS)
  write.csv(qcdf, file.path(out_root, "scVI_counts", paste0(s, "_qc_metrics.csv")),
            row.names = FALSE)

  # ...but the counts CSV holds only the QC-PASSING spots.  The original notebook
  # subset the object before writing, so dataset/ST/scVI_counts/IU_PDA_T4.csv has
  # 3,587 barcode columns for a 3,621-spot sample.  Emitting all spots here would
  # silently feed scVI a different (larger, lower-quality) training set than the
  # one every validated result was produced on.
  n_raw <- ncol(cm)
  cm <- cm[, qcdf$pass_qc, drop = FALSE]
  msg("  QC filter: ", ncol(cm), "/", n_raw, " spots retained (",
      n_raw - ncol(cm), " dropped, ",
      sprintf("%.1f%%", 100 * (n_raw - ncol(cm)) / n_raw), ")")
  write_counts_csv(cm, file.path(out_root, "scVI_counts", paste0(s, ".csv")))

  # ---- coordinates
  im <- img_for(s)
  if (!is.na(im)) {
    co <- obj@images[[im]]@coordinates[cells, , drop = FALSE]
    co$barcode <- rownames(co)
    write.csv(co[, c("barcode", "row", "col", "imagerow", "imagecol")],
              file.path(out_root, "coords", paste0(s, "_coords.csv")), row.names = FALSE)
  } else {
    msg("  WARNING no image matched for ", s, " -- coordinates not written")
  }

  # ---- published RCTD (the cell modality)
  if ("rctd_fullfinal" %in% Assays(sub))
    write_spot_matrix(sub, s, "rctd_fullfinal", "rctd", "_rctd_fullfinal.csv")

  # ---- Bagaev functional signatures
  if ("fges" %in% Assays(sub)) write_spot_matrix(sub, s, "fges", "fges", "_fges.csv")

  # gzip level 1: these slim objects are re-read constantly downstream and the
  # default level-6 compression dominates per-sample wall clock for ~5% of the size.
  con <- gzfile(file.path(out_root, "ST", paste0(s, ".rds")), "wb", compression = 1)
  saveRDS(sub, con); close(con)
  summary_rows[[s]] <- data.frame(
    sample = s, n_spots = ncol(sub), n_spots_pass_qc = sum(qcdf$pass_qc),
    n_genes = nrow(cm),
    median_nFeature = median(nGene), median_nCount = median(nUMI),
    image = ifelse(is.na(im), "MISSING", im))
  msg("  ", ncol(sub), " spots | median genes ", median(nGene),
      " | median counts ", median(nUMI))
  rm(sub, cm); gc()
}

summ <- do.call(rbind, summary_rows); rownames(summ) <- NULL
write.csv(summ, file.path(out_root, "split_summary.csv"), row.names = FALSE)
msg("DONE -> ", out_root)
print(summ)

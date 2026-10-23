# Valida src/pruebas.py::delong contra pROC::roc.test(paired = TRUE, method = "delong").
#
# Sin argumentos: genera 6 casos sinteticos (tamanos distintos, puntajes con empates y sin ellos, AUC parecidos y
# distintos) y escribe los datos en tests/datos/delong_casos.csv y lo que da pROC en tests/datos/delong_proc.csv.
# pytest compara contra esos dos archivos, asi el CI no necesita R.
# Con un argumento (un CSV con columnas y, s1, s2): escribe una fila con lo que da pROC en ese caso, para validar en
# datos reales (resultados/pruebas/validacion_delong.csv lo arma src/comparar_pruebas.py).
# Uso: "C:\Program Files\R\R-4.5.0\bin\Rscript.exe" src/validar_delong.R [caso.csv salida.csv]
suppressPackageStartupMessages(library(pROC))

prueba <- function(y, s1, s2) {
  r1 <- roc(y, s1, levels = c(0, 1), direction = "<", quiet = TRUE)
  r2 <- roc(y, s2, levels = c(0, 1), direction = "<", quiet = TRUE)
  t <- roc.test(r1, r2, paired = TRUE, method = "delong")
  data.frame(AUC_1 = as.numeric(auc(r1)), AUC_2 = as.numeric(auc(r2)), z = unname(t$statistic), p = t$p.value)
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 2) {
  d <- read.csv(args[1])
  write.csv(prueba(d$y, d$s1, d$s2), args[2], row.names = FALSE)
  quit(save = "no")
}

set.seed(2026)
casos <- list(); res <- list()
config <- list(c(200, 0.8, 1), c(500, 0.5, 2), c(1000, 1.0, 0), c(300, 0.3, 1), c(2000, 0.6, 2), c(80, 1.2, 1))
for (k in seq_along(config)) {
  n <- config[[k]][1]; efecto <- config[[k]][2]; redondeo <- config[[k]][3]
  y <- rbinom(n, 1, 0.35)
  base <- rnorm(n)
  s1 <- base + efecto * y + rnorm(n, sd = 0.8)
  s2 <- 0.7 * base + 0.8 * efecto * y + rnorm(n, sd = 1)
  if (redondeo > 0) { s1 <- round(s1, redondeo - 1); s2 <- round(s2, redondeo - 1) }   # fuerza empates
  casos[[k]] <- data.frame(caso = k, y = y, s1 = s1, s2 = s2)
  res[[k]] <- cbind(caso = k, prueba(y, s1, s2))
}
dir.create("tests/datos", showWarnings = FALSE, recursive = TRUE)
write.csv(do.call(rbind, casos), "tests/datos/delong_casos.csv", row.names = FALSE)
write.csv(do.call(rbind, res), "tests/datos/delong_proc.csv", row.names = FALSE)
print(do.call(rbind, res))

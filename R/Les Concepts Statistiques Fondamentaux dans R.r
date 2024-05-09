library(readr)
library(tidyverse)

data <- read.csv("Cross-sec_full.csv", stringsAsFactors = FALSE, na.strings = c(" ","."))

hist(x = data$V12.aosi.total_score_1_18, xlab ="Score Total de l'AOSI", main= "Histogramme du Score Total de l'AOSI à 12 mois")

data_HR_ASD <- data  %>% filter(GROUP =="HR_ASD")

hist(data_HR_ASD$V12.aosi.total_score_1_18, xlab ="Score Total de l'AOSI", main= "Histogramme du Score Total de l'AOSI à 12 mois pour le groupe 'High Risk: ASD'")

# Générer une séquence de valeurs x
x <- seq(-10, 10, length.out=100)

# Calculer les densités de probabilité normales pour chaque valeur de x
y1 <- dnorm(x, mean=0, sd=1)
y2 <- dnorm(x, mean=0, sd=4)

# Tracer la distribution normale
plot(x, y1, type="l", lwd=2, col="blue", xlab="Valeurs", ylab="Densité de probabilité", main="Distribution Normale de variance 1")
plot(x, y2, type="l", lwd=2, col="blue", xlab="Valeurs", ylab="Densité de probabilité", main="Distribution Normale de variance 4")


# La moyenne
mean(data$V12.aosi.total_score_1_18, na.rm = TRUE)

# La variance
var(data$V12.aosi.total_score_1_18, na.rm = TRUE)

# La médiane
median(data$V12.aosi.total_score_1_18, na.rm = TRUE)

# quatile
quantile(data$V12.aosi.total_score_1_18, na.rm = TRUE)

data_petite <- data %>% 
    select(V12.aosi.total_score_1_18, V06.aosi.total_score_1_18,
         V12.aosi.Candidate_Age)


head(data_petite)

summary(data_petite)

library(Hmisc)
describe(data_petite)

data_simul <- list()

for(i in 1:5){
    data_simul[[i]] <- rnorm(100, 0, 1) # simuler 100 obs de moyenne 0 et variance 1 
}

lapply(data_simul, mean)

mean(data$V12.aosi.total_score_1_18, na.rm = TRUE)

resultats <- t.test(data$V12.aosi.total_score_1_18, conf.level = 0.95)
resultats

# Extraction de l'intervalle de confiance
int_conf <- resultats$conf.int
int_conf

resultats$statistic

resultats$parameter

resultats$p.value

resultats$estimate

resultats$null.value

resultats$stderr

# écart-type
ecart_type <- sd(data$V12.aosi.total_score_1_18, na.rm = TRUE)
ecart_type

# écart-type
sqrt(var(data$V12.aosi.total_score_1_18, na.rm = TRUE))

# SE = s/sqrt(n)
ecart_type / sqrt(length(data$V12.aosi.total_score_1_18))

# Intervalle de confiance manuellement : Z_score = 1.96 pour 95% niveau de confidence
lim_sup <- mean(data$V12.aosi.total_score_1_18, na.rm = TRUE) + 1.96 * ecart_type / sqrt(length(data$V12.aosi.total_score_1_18))
lim_sup

lim_inf <- mean(data$V12.aosi.total_score_1_18, na.rm = TRUE) - 1.96 * ecart_type / sqrt(length(data$V12.aosi.total_score_1_18))
lim_inf

resultats$alternative

resultats$method

resultats$data.name

t.test(data$V12.aosi.total_score_1_18)

t.test(data$V12.aosi.total_score_1_18)

head(data)

data_high_risk <- data  %>% 
    filter((GROUP == "HR_ASD") | (GROUP =="HR_neg"))

t.test(data = data_high_risk, V12.aosi.total_score_1_18~GROUP)

unique(data$GROUP)

objet_aov <- aov(V12.aosi.total_score_1_18~GROUP, data= data)
objet_aov

summary(objet_aov)

TukeyHSD(objet_aov)

library(ggplot2)

?geom_histogram

ggplot(data = data, aes(x = V12.aosi.total_score_1_18, fill = GROUP)) +
  geom_histogram()+
  facet_wrap(~ GROUP, ncol = 2)



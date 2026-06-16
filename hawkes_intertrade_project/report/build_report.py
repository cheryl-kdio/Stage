from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Preformatted,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

OUT = "/mnt/data/hawkes_intertrade_report.pdf"

pdfmetrics.registerFont(TTFont("DejaVu", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))

styles = getSampleStyleSheet()
for name in list(styles.byName):
    styles[name].fontName = "DejaVu"

styles.add(ParagraphStyle(
    name="TitleCustom",
    parent=styles["Title"],
    fontName="DejaVu-Bold",
    fontSize=20,
    leading=25,
    alignment=TA_CENTER,
    spaceAfter=18,
))
styles.add(ParagraphStyle(
    name="H1Custom",
    parent=styles["Heading1"],
    fontName="DejaVu-Bold",
    fontSize=15,
    leading=20,
    spaceBefore=12,
    spaceAfter=8,
))
styles.add(ParagraphStyle(
    name="H2Custom",
    parent=styles["Heading2"],
    fontName="DejaVu-Bold",
    fontSize=12,
    leading=16,
    spaceBefore=10,
    spaceAfter=6,
))
styles.add(ParagraphStyle(
    name="BodyCustom",
    parent=styles["BodyText"],
    fontName="DejaVu",
    fontSize=9.5,
    leading=13.5,
    alignment=TA_LEFT,
    spaceAfter=6,
))
styles.add(ParagraphStyle(
    name="SmallCustom",
    parent=styles["BodyText"],
    fontName="DejaVu",
    fontSize=8.2,
    leading=11,
    spaceAfter=4,
))
styles.add(ParagraphStyle(
    name="Formula",
    parent=styles["BodyText"],
    fontName="DejaVu",
    fontSize=9,
    leading=12,
    leftIndent=12,
    backColor=colors.whitesmoke,
    borderPadding=6,
    spaceBefore=4,
    spaceAfter=8,
))
styles.add(ParagraphStyle(
    name="CodeCustom",
    parent=styles["Code"],
    fontName="Courier",
    fontSize=7.6,
    leading=9.4,
    leftIndent=0,
    spaceBefore=4,
    spaceAfter=8,
))


def P(text, style="BodyCustom"):
    return Paragraph(text, styles[style])


def bullets(items):
    story = []
    for item in items:
        story.append(P("- " + item, "BodyCustom"))
    return story


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DejaVu", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(1.5 * cm, 1.0 * cm, "Hawkes intertrades - rapport methodologique")
    canvas.drawRightString(A4[0] - 1.5 * cm, 1.0 * cm, f"Page {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(
    OUT,
    pagesize=A4,
    rightMargin=1.55 * cm,
    leftMargin=1.55 * cm,
    topMargin=1.45 * cm,
    bottomMargin=1.45 * cm,
)

story = []
story.append(P("Analyse de l'impact du volume sur les durees intertrades", "TitleCustom"))
story.append(P("Projet Python et plan de recherche quantitatif base sur processus de Hawkes", "H2Custom"))
story.append(Spacer(1, 8))
story.append(P("Ce rapport formalise l'objectif, le modele, la methodologie et le deroule pratique pour analyser si le volume des trades influence le temps d'attente avant les trades suivants, puis tester si cette information peut etre transformee en signal de trading.", "BodyCustom"))
story.append(P("Livrables associes: package Python <b>hawkes_intertrade</b>, exemples synthetiques, tests de fumee, squelette de signal et backtest exploratoire.", "BodyCustom"))
story.append(Spacer(1, 10))

story.append(P("1. Definition du processus de Hawkes", "H1Custom"))
story.append(P("Un processus de Hawkes est un processus ponctuel auto-excitant: l'arrivee d'un evenement augmente temporairement l'intensite d'arrivee des evenements futurs. En version multivariee, chaque dimension peut exciter les autres dimensions.", "BodyCustom"))
story.append(P("Modele multivarie avec noyau exponentiel:", "BodyCustom"))
story.append(P("lambda_i(t) = mu_i(t) + sum_j sum_{t_k^j < t} alpha_{ij} beta_{ij} exp(-beta_{ij}(t - t_k^j))", "Formula"))
story.append(P("Convention retenue: alpha_{ij} mesure l'excitation de la dimension source j vers la dimension cible i. Le parametre beta_{ij} controle la vitesse de decroissance. Plus beta est grand, plus l'effet est bref.", "BodyCustom"))
story.append(P("La stationnarite du modele exponentiel est souvent controlee par le rayon spectral de la matrice alpha. Dans cette parametrisation, une condition usuelle est rho(alpha) < 1.", "BodyCustom"))

story.append(P("2. Objectif de l'etude", "H1Custom"))
story.extend(bullets([
    "Mesurer si le volume d'un trade modifie l'intensite des trades futurs.",
    "Comparer l'information du volume aux effets de clustering deja captures par les durees passees.",
    "Separer l'effet activite - quand le prochain trade arrive - de l'effet directionnel - prix up ou down.",
    "Construire un signal de trading seulement si l'effet est robuste hors echantillon et apres couts.",
]))
story.append(P("Le point central n'est donc pas seulement de predire la duree intertrade moyenne. Il est preferable de predire une intensite conditionnelle, puis d'en deduire une probabilite d'activite prochaine.", "BodyCustom"))
story.append(P("P(tau <= h | F_t) = 1 - exp(- integral_t^{t+h} lambda(s | F_t) ds)", "Formula"))

story.append(P("3. Donnees d'entree", "H1Custom"))
story.append(P("Le projet part d'une table de trades au format minimal:", "BodyCustom"))
story.append(Preformatted("timestamp, price, volume", styles["CodeCustom"]))
story.append(P("Une colonne side est optionnelle. Si elle est absente, le package peut inferer un side buy/sell par tick rule. Cette inference est utile pour prototyper, mais elle n'est pas equivalente a une classification de trades basee sur carnet d'ordres et doit etre traitee avec prudence.", "BodyCustom"))
story.append(P("Features construites:", "BodyCustom"))
story.extend(bullets([
    "t: temps en secondes depuis le premier trade.",
    "duration et next_duration: durees intertrades.",
    "log_volume et log_volume_z: volume transforme et standardise.",
    "volume_bucket: bucket de volume par quantiles.",
    "side_sign: buy/sell, fourni ou infere.",
    "price_up_event et price_down_event: evenements de prix directionnels.",
]))

story.append(P("4. Methodologie proposee", "H1Custom"))
story.append(P("La methodologie est volontairement progressive. Elle evite de construire directement un modele Hawkes tres riche avant d'avoir prouve que le volume ajoute de l'information predictive.", "BodyCustom"))

table_data = [
    ["Etape", "Modele / test", "Question traitee"],
    ["A", "Analyse descriptive", "Les gros volumes sont-ils suivis de durees plus courtes ou plus longues?"],
    ["B", "Log-ACD + volume", "Le volume predit-il la duree suivante apres controle des durees passees?"],
    ["C", "Hawkes univarie", "Les trades arrivent-ils en clusters auto-excites?"],
    ["D", "Hawkes volume-buckets", "Les gros volumes excitent-ils davantage l'activite future?"],
    ["E", "Hawkes buy/sell + volume", "Le volume agit-il differemment selon le sens du trade?"],
    ["F", "Hawkes + price up/down", "L'activite predite contient-elle de l'information directionnelle?"],
    ["G", "Signal + backtest", "Le signal survit-il aux couts, spread, latence et slippage?"],
]

table = Table(table_data, colWidths=[1.3 * cm, 4.1 * cm, 11.0 * cm])
table.setStyle(TableStyle([
    ("FONTNAME", (0, 0), (-1, 0), "DejaVu-Bold"),
    ("FONTNAME", (0, 1), (-1, -1), "DejaVu"),
    ("FONTSIZE", (0, 0), (-1, -1), 7.6),
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAECEF")),
    ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
story.append(table)
story.append(Spacer(1, 8))

story.append(P("5. Modeles implementes dans le projet", "H1Custom"))
story.append(P("5.1 Benchmark Log-ACD avec volume", "H2Custom"))
story.append(P("Le modele Log-ACD sert de benchmark pour les durees. Il modelise directement l'esperance conditionnelle de la duree intertrade suivante.", "BodyCustom"))
story.append(P("duration_t = psi_t epsilon_t, epsilon_t ~ Exp(1)", "Formula"))
story.append(P("log psi_t = omega + a log(duration_{t-1}) + b log(psi_{t-1}) + gamma mark_{t-1}", "Formula"))
story.append(P("Si gamma est significatif et robuste hors echantillon, le volume transporte de l'information sur la duree suivante.", "BodyCustom"))

story.append(P("5.2 Hawkes exponentiel multivarie", "H2Custom"))
story.append(P("Deux estimateurs sont fournis: un avec decays fixes et un avec decays estimes. La version a decays fixes est plus stable et plus proche d'un premier benchmark. La version a decays libres est plus flexible mais non convexe et sensible a l'initialisation.", "BodyCustom"))
story.extend(bullets([
    "HawkesExpFixedDecayMLE: estime mu et alpha avec beta fixe.",
    "HawkesExpFreeDecayMLE: estime mu, alpha et beta avec multi-start.",
    "Les evenements simultanes sont geres comme des sauts multiples au meme timestamp.",
    "Le rayon spectral de alpha est calcule apres estimation pour diagnostiquer la stabilite.",
]))

story.append(P("5.3 Volume par buckets", "H2Custom"))
story.append(P("Le choix recommande pour commencer est de categoriser le volume en quantiles. Au lieu d'estimer un effet continu fragile, on construit des dimensions d'evenements par bucket de volume.", "BodyCustom"))
story.append(P("Exemple de streams: B_q0, B_q1, B_q2, S_q0, S_q1, S_q2, P_UP, P_DOWN.", "Formula"))
story.append(P("Cette representation permet de lire directement quelles categories de volume excitent les trades futurs ou les mouvements de prix.", "BodyCustom"))

story.append(PageBreak())
story.append(P("6. Signal d'activite et signal directionnel", "H1Custom"))
story.append(P("Une fois le modele multivarie estime, on peut convertir les intensites en quantites exploitables sur un horizon h.", "BodyCustom"))
story.append(P("I_i(t,h) = integral_t^{t+h} lambda_i(s | F_t) ds", "Formula"))
story.append(P("Pour les dimensions de prix, on calcule:", "BodyCustom"))
story.append(P("activity(t,h) = I_UP(t,h) + I_DOWN(t,h)", "Formula"))
story.append(P("direction(t,h) = [I_UP(t,h) - I_DOWN(t,h)] / [I_UP(t,h) + I_DOWN(t,h) + eps]", "Formula"))
story.append(P("Le signal final du squelette de strategie est:", "BodyCustom"))
story.append(P("signal(t,h) = activity(t,h) * direction(t,h)", "Formula"))
story.append(P("Interpretation: on ne trade pas seulement parce que l'activite est forte. On cherche une combinaison entre activite imminente et desequilibre directionnel.", "BodyCustom"))

story.append(P("7. Validation statistique", "H1Custom"))
story.extend(bullets([
    "Comparer la log-vraisemblance out-of-sample entre Poisson saisonnier, Hawkes sans volume et Hawkes avec volume.",
    "Tester les residus par time-rescaling: les increments compenses doivent ressembler a des Exp(1).",
    "Verifier que le signal directionnel predit effectivement les variations de prix futures, pas seulement l'activite.",
    "Faire une validation walk-forward pour limiter le look-ahead et mesurer la degradation hors echantillon.",
]))
story.append(P("Le volume peut tres bien predire l'activite sans predire la direction. C'est pour cette raison que l'ajout des dimensions P_UP et P_DOWN est une etape importante avant tout backtest.", "BodyCustom"))

story.append(P("8. Backtest et prudence execution", "H1Custom"))
story.append(P("Le projet inclut un backtest minimal, mais celui-ci n'est qu'un squelette. Un backtest haute frequence credible doit inclure les couts et contraintes d'execution.", "BodyCustom"))
story.extend(bullets([
    "Spread et frais explicites.",
    "Latence entre calcul du signal et envoi de l'ordre.",
    "Slippage et impact de marche.",
    "Execution au bid/ask, jamais gratuitement au mid.",
    "File d'attente et probabilite de fill pour une strategie maker.",
    "Controle strict de l'information disponible au moment de la decision.",
]))
story.append(P("La regle fournie dans le code est volontairement simple: acheter si direction > seuil et activite > seuil, vendre si direction < -seuil et activite > seuil, sinon ne rien faire.", "BodyCustom"))

story.append(P("9. Deroule pratique recommande", "H1Custom"))
story.extend(bullets([
    "Importer une journee ou un actif liquide et convertir timestamp, price, volume en DataFrame propre.",
    "Nettoyer les doublons, verifier l'ordre temporel et retirer les periodes sans marche actif.",
    "Retirer ou modeliser la saisonnalite intraday de l'intensite.",
    "Analyser E[next_duration | volume_bucket] et les correlations log-duration/log-volume.",
    "Estimer le Log-ACD avec volume et verifier le signe de gamma.",
    "Construire les streams volume-buckets, puis buy/sell + volume-buckets + price up/down.",
    "Estimer Hawkes a decays fixes, puis seulement ensuite tester les decays libres.",
    "Comparer les performances out-of-sample et les residus.",
    "Construire le signal direction/activite et lancer un backtest walk-forward avec couts.",
]))

story.append(P("10. Architecture du projet Python", "H1Custom"))
story.append(Preformatted("""hawkes_intertrade/
  acd.py        LogACDVolumeMLE
  data.py       features et construction des streams
  hawkes.py     HawkesExpFixedDecayMLE, HawkesExpFreeDecayMLE
  signal.py     intensites integrees, scores activite/direction
  backtest.py   squelette de backtest exploratoire
examples/
  synthetic_fit.py
  workflow_from_trades.py
tests/
  test_smoke.py""", styles["CodeCustom"]))
story.append(P("Les exemples sont synthetiques afin que le projet soit executable sans donnees proprietaires. Sur donnees reelles, le point le plus important sera la qualite du timestamping, du sens de trade et de la mesure du prix executable.", "BodyCustom"))

story.append(P("11. Limites et extensions", "H1Custom"))
story.extend(bullets([
    "Les decays libres rendent la vraisemblance non convexe. Le multi-start est indispensable.",
    "Un Hawkes continu suppose des timestamps d'evenements. Des donnees agregees par bars demandent un modele discret ou une approximation explicite.",
    "Le modele actuel n'inclut pas de baseline intraday non parametrique. C'est une extension prioritaire pour des donnees reelles.",
    "La strategie fournie ne modelise pas la microstructure d'execution. Elle doit etre remplacee par un simulateur bid/ask si l'objectif est un test de PnL credible.",
    "Une extension naturelle consiste a ajouter imbalance du carnet, spread, volatilite locale et regimes intraday comme covariables.",
]))

story.append(P("References", "H1Custom"))
refs = [
    "Hawkes, A. G. (1971). Spectra of some self-exciting and mutually-exciting point processes. Biometrika, 58(1), 83-90.",
    "Engle, R. F. and Russell, J. R. (1998). Autoregressive Conditional Duration: A New Model for Irregularly Spaced Transaction Data. Econometrica, 66(5), 1127-1162.",
    "Bacry, E., Mastromatteo, I. and Muzy, J.-F. (2015). Hawkes Processes in Finance. Market Microstructure and Liquidity.",
    "Bacry, E. and Muzy, J.-F. (2013). Hawkes model for price and trades high-frequency dynamics.",
    "tick documentation: tick.hawkes.HawkesExpKern, learner for exponential Hawkes kernels with fixed decays.",
]
for ref in refs:
    story.append(P("- " + ref, "SmallCustom"))

story.append(Spacer(1, 6))
story.append(P("Conclusion: la bonne sequence est de valider d'abord que le volume explique les durees, puis qu'il ameliore l'intensite Hawkes hors echantillon, et seulement ensuite de tester si l'intensite directionnelle price up/down peut produire un alpha apres couts.", "BodyCustom"))

doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT)

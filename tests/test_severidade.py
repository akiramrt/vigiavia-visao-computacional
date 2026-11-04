import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vigiavia import severidade as sev  # noqa: E402


def test_area_relativa():
    assert sev.area_relativa(0, 0, 50, 50, 100, 100) == 0.25


def test_mesma_caixa_mais_perto_do_horizonte_e_mais_severa():
    # mesma área em pixels: a caixa mais alta na imagem (mais distante) tem índice maior
    perto = sev.indice_severidade(400, 600, 480, 680, 1000, 700)
    longe = sev.indice_severidade(400, 300, 480, 380, 1000, 700)
    assert longe > perto


def test_indice_limitado_a_um():
    assert sev.indice_severidade(0, 0, 1000, 700, 1000, 700) <= 1.0


def test_classificacao_respeita_limiares():
    lim = (0.01, 0.05)
    assert sev.classificar(0.001, lim) == "baixa"
    assert sev.classificar(0.02, lim) == "media"
    assert sev.classificar(0.2, lim) == "alta"

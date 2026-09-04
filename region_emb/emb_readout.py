import json
from typing import Iterable, Set
from typing import Dict, List, Tuple, Optional, Union

import torch
import torch.nn.functional as F


class CityEmbLib:
    """
    payload:
      payload["keys"] : List[Tuple[str,int]]    # gid -> (county5, local_idx)
      payload["emb"]  : FloatTensor [total_zones, d]
      payload["dim"]  : int
      payload["normalize"] : bool (optional)

    city_geoids:
      Dict[str, List[str]]  # county_geoid(5位) -> list of tract_geoids(6位) in local order
    """

    def __init__(self, emb_path: str, geoid_path: str, device: Optional[str] = None):
        payload = torch.load(emb_path, map_location="cpu")
        self.payload = payload

        self.raw_keys: List[Tuple[str, int]] = payload["keys"]   # ✅ list of (county5, local_idx)
        self.emb: torch.Tensor = payload["emb"]
        self.dim: int = int(payload.get("dim", self.emb.shape[1]))
        self.is_normalized: bool = bool(payload.get("normalize", False))

        with open(geoid_path, encoding="utf-8") as f:
            self.city_geoids: Dict[str, List[str]] = json.load(f)

        # device
        self.device = device
        if self.device is not None:
            self.emb = self.emb.to(self.device)

        # ✅ 1) 先把 list keys 变成 county -> [gid...]
        self.global_keys: Dict[str, List[int]] = self._normalize_keys(self.raw_keys)

        # ✅ 2) 建索引 tract11 <-> gid
        self.tract11_to_gid: Dict[str, int] = {}
        self.gid_to_tract11: List[str] = [""] * self.emb.shape[0]
        self._build_index_from_raw_keys()

        # ✅ 3) 检索用归一化 embedding
        self.emb_norm = self._get_emb_for_search()

    # --------------------------
    # internal utils
    # --------------------------
    @staticmethod
    def _pad_county(county: Union[str, int]) -> str:
        return str(county).zfill(5)

    @staticmethod
    def _pad_tract(tract: Union[str, int]) -> str:
        return str(tract).zfill(6)

    @classmethod
    def make_tract11(cls, county5: Union[str, int], tract6: Union[str, int]) -> str:
        return cls._pad_county(county5) + cls._pad_tract(tract6)

    def _normalize_keys(self, raw_keys: List[Tuple[str, int]]) -> Dict[str, List[int]]:
        """
        raw_keys: gid -> (county5, local_idx)
        返回: county5 -> [gid0, gid1, ...] (按 gid 顺序，也就是 emb 的顺序)
        """
        out: Dict[str, List[int]] = {}
        for gid, (county, local_idx) in enumerate(raw_keys):
            county5 = self._pad_county(county)
            out.setdefault(county5, []).append(int(gid))
        return out

    def _build_index_from_raw_keys(self):
        """
        用 raw_keys 逐行建立 tract11 <-> gid 映射：
          gid -> (county5, local_idx) -> tract6 -> tract11
        比你原来的 “长度对齐” 更可靠（不会依赖 global_keys 的顺序假设）。
        """
        for gid, (county, local_idx) in enumerate(self.raw_keys):
            county5 = self._pad_county(county)

            if county5 not in self.city_geoids:
                # 这个 county 在 county2tract 里没有（可能文件不全）
                continue

            tract_list = self.city_geoids[county5]
            li = int(local_idx)
            if li < 0 or li >= len(tract_list):
                raise IndexError(
                    f"[CityEmbLib] local_idx out of range: county={county5}, "
                    f"local_idx={li}, but tract_list has len={len(tract_list)}"
                )

            tract6 = tract_list[li]
            tract11 = self.make_tract11(county5, tract6)

            self.tract11_to_gid[tract11] = int(gid)
            self.gid_to_tract11[int(gid)] = tract11

    def _get_emb_for_search(self) -> torch.Tensor:
        emb = self.emb
        # ✅ 去掉全局公共分量（让相似度更有区分度）
        emb = emb - emb.mean(dim=0, keepdim=True)
        if not self.is_normalized:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb

    def _get_gid(self, tract11: Union[str, int]) -> int:
        tract11 = str(tract11).zfill(11)
        if tract11 not in self.tract11_to_gid:
            raise KeyError(f"[CityEmbLib] tract11={tract11} not found in embedding index.")
        return self.tract11_to_gid[tract11]

    # --------------------------
    # 1) city -> all zone embeddings
    # --------------------------
    def get_city_zone_emb(self, county_geoid: Union[str, int], return_ids: bool = True):
        county5 = self._pad_county(county_geoid)
        if county5 not in self.global_keys:
            raise KeyError(f"[CityEmbLib] county={county5} not embedded.")

        gids = self.global_keys[county5]
        idx = torch.tensor(gids, dtype=torch.long, device=self.emb.device)
        city_embs = self.emb.index_select(0, idx)

        if not return_ids:
            return city_embs

        tract11_list = [self.gid_to_tract11[g] for g in gids]
        return city_embs, tract11_list

    # --------------------------
    # 2) tract11 -> one zone embedding
    # --------------------------
    def get_zone_emb(self, tract11: Union[str, int]) -> torch.Tensor:
        gid = self._get_gid(tract11)
        return self.emb[gid]

    # --------------------------
    # 3) similarity search: tract11 -> topk similar tract11
    # --------------------------
    @torch.no_grad()
    def topk_similar_tracts(
        self,
        tract11: Union[str, int],
        k: int = 10,
        exclude_self: bool = True,
        return_scores: bool = False,
        restrict_to_county: Optional[Union[str, int]] = None,
    ):
        q_gid = self._get_gid(tract11)
        q = self.emb_norm[q_gid]  # [d]

        if restrict_to_county is not None:
            county5 = self._pad_county(restrict_to_county)
            if county5 not in self.global_keys:
                raise KeyError(f"[CityEmbLib] restrict county={county5} not embedded.")
            cand_gids = self.global_keys[county5]
            cand = self.emb_norm.index_select(
                0, torch.tensor(cand_gids, dtype=torch.long, device=self.emb_norm.device)
            )
            scores = cand @ q
            topn = min(k + (1 if exclude_self else 0), scores.numel())
            vals, pos = torch.topk(scores, k=topn, largest=True)
            gids = [cand_gids[int(i)] for i in pos.tolist()]
            vals = vals.tolist()
        else:
            scores = self.emb_norm @ q
            topn = min(k + (1 if exclude_self else 0), scores.numel())
            vals, gids = torch.topk(scores, k=topn, largest=True)
            gids = gids.tolist()
            vals = vals.tolist()

        out = []
        for gid, s in zip(gids, vals):
            if exclude_self and int(gid) == int(q_gid):
                continue
            tid = self.gid_to_tract11[int(gid)]
            if tid == "":
                continue
            out.append((tid, float(s)) if return_scores else tid)
            if len(out) >= k:
                break
        return out

    @torch.no_grad()
    def topk_similar_tracts_in_set(
            self,
            tract11: Union[str, int],
            allowed_tracts: Iterable[str],
            k: int = 10,
            exclude_self: bool = True,
            return_scores: bool = False,
    ):
        """
        Find top-k similar tracts but restrict candidates to `allowed_tracts` (tract11 list/set).
        This is what you want for "only search among tracts that have shapes in ShapeMem".
        """
        # make a fast set
        allowed_set: Set[str] = set(str(t).zfill(11) for t in allowed_tracts)

        q_gid = self._get_gid(tract11)
        q = self.emb_norm[q_gid]  # [d]

        # convert allowed tract11 -> gids (skip those not embedded)
        cand_gids: List[int] = []
        for tid in allowed_set:
            gid = self.tract11_to_gid.get(tid)
            if gid is not None:
                cand_gids.append(int(gid))

        if len(cand_gids) == 0:
            return []

        cand = self.emb_norm.index_select(
            0, torch.tensor(cand_gids, dtype=torch.long, device=self.emb_norm.device)
        )  # [C, d]

        scores = cand @ q  # [C]
        topn = min(k + (1 if exclude_self else 0), scores.numel())
        vals, pos = torch.topk(scores, k=topn, largest=True)

        out = []
        for p, s in zip(pos.tolist(), vals.tolist()):
            gid = cand_gids[int(p)]
            if exclude_self and gid == int(q_gid):
                continue
            tid = self.gid_to_tract11[gid]
            if tid == "":
                continue
            out.append((tid, float(s)) if return_scores else tid)
            if len(out) >= k:
                break
        return out


if __name__ == '__main__':
    CityEmb = CityEmbLib(
        emb_path='../ckpts/zone_embeddings.pt',
        geoid_path="../county2tract.json",
        device=None
    )

    city_id = "18077"
    city_embs, tract11s = CityEmb.get_city_zone_emb(city_id)
    print(city_embs.shape, tract11s[:3])

    tract_id = "18077966300"
    tract_z = CityEmb.get_zone_emb(tract_id)
    print(tract_z.shape)

    near_tract_ids = CityEmb.topk_similar_tracts(tract_id, k=10, return_scores=True)
    print(near_tract_ids)

    near_in_county = CityEmb.topk_similar_tracts(tract_id, k=10, return_scores=True, restrict_to_county="18077")
    print(near_in_county)

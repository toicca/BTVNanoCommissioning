import collections, awkward as ak, numpy as np
import os
import uproot
from coffea import processor
from coffea.analysis_tools import Weights
from BTVNanoCommissioning.utils.correction import (
    load_lumi,
    load_SF,
    muSFs,
    eleSFs,
    puwei,
    btagSFs,
    JME_shifts,
    Roccor_shifts,
)

from BTVNanoCommissioning.helpers.func import (
    flatten,
    update,
    dump_lumi,
)
from BTVNanoCommissioning.helpers.update_branch import missing_branch
from BTVNanoCommissioning.utils.histogrammer import histogrammer
from BTVNanoCommissioning.utils.array_writer import array_writer
from BTVNanoCommissioning.utils.selection import (
    jet_id,
    mu_idiso,
    ele_mvatightid,
    softmu_mask,
)


class NanoProcessor(processor.ProcessorABC):
    def __init__(
        self,
        year="2022",
        campaign="Summer22Run3",
        name="",
        isSyst=False,
        isArray=False,
        noHist=False,
        chunksize=75000,
    ):
        self._year = year
        self._campaign = campaign
        self.name = name
        self.isSyst = isSyst
        self.isArray = isArray
        self.noHist = noHist
        self.lumiMask = load_lumi(self._campaign)
        self.chunksize = chunksize
        ## Load corrections
        self.SF_map = load_SF(self._campaign)

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        isRealData = not hasattr(events, "genWeight")
        dataset = events.metadata["dataset"]
        events = missing_branch(events)
        shifts = []
        if "JME" in self.SF_map.keys():
            syst_JERC = self.isSyst
            if self.isSyst == "JERC_split":
                syst_JERC = "split"
            shifts = JME_shifts(
                shifts, self.SF_map, events, self._campaign, isRealData, syst_JERC
            )
        else:
            if int(self._year) < 2020:
                shifts = [
                    ({"Jet": events.Jet, "MET": events.MET, "Muon": events.Muon}, None)
                ]
            else:
                shifts = [
                    (
                        {
                            "Jet": events.Jet,
                            "MET": events.PuppiMET,
                            "Muon": events.Muon,
                        },
                        None,
                    )
                ]
        if "roccor" in self.SF_map.keys():
            shifts = Roccor_shifts(shifts, self.SF_map, events, isRealData, False)
        else:
            shifts[0][0]["Muon"] = events.Muon

        return processor.accumulate(
            self.process_shift(update(events, collections), name)
            for collections, name in shifts
        )

    def process_shift(self, events, shift_name):
        dataset = events.metadata["dataset"]
        isRealData = not hasattr(events, "genWeight")
        _hist_event_dict = (
            {"": None} if self.noHist else histogrammer(events, "emctag_ttdilep_sf")
        )

        output = {
            "sumw": processor.defaultdict_accumulator(float),
            **_hist_event_dict,
        }

        if shift_name is None:
            if isRealData:
                output["sumw"] = len(events)
            else:
                output["sumw"] = ak.sum(events.genWeight)
        ####################
        #    Selections    #
        ####################
        ## Lumimask
        req_lumi = np.ones(len(events), dtype="bool")
        if isRealData:
            req_lumi = self.lumiMask(events.run, events.luminosityBlock)
        # only dump for nominal case
        if shift_name is None:
            output = dump_lumi(events[req_lumi], output)

        ## HLT
        trigger_he = [
            "Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL",
            "Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
        ]
        trigger_hm = [
            "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL",
            "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
        ]

        checkHLT = ak.Array(
            [hasattr(events.HLT, _trig) for _trig in trigger_he + trigger_hm]
        )
        if ak.all(checkHLT == False):
            raise ValueError(
                "HLT paths:", trigger_he + trigger_hm, " are all invalid in", dataset
            )
        elif ak.any(checkHLT == False):
            print(
                np.array(trigger_he + trigger_hm)[~checkHLT], " not exist in", dataset
            )
        trig_arr_ele = [
            events.HLT[_trig] for _trig in trigger_he if hasattr(events.HLT, _trig)
        ]
        req_trig_ele = np.zeros(len(events), dtype="bool")
        for t in trig_arr_ele:
            req_trig_ele = req_trig_ele | t
        trig_arr_mu = [
            events.HLT[_trig] for _trig in trigger_hm if hasattr(events.HLT, _trig)
        ]
        req_trig_mu = np.zeros(len(events), dtype="bool")
        for t in trig_arr_mu:
            req_trig_mu = req_trig_mu | t

        ## Muon cuts
        iso_muon_mu = events.Muon[
            (events.Muon.pt > 24) & mu_idiso(events, self._campaign)
        ]
        iso_muon_ele = events.Muon[
            (events.Muon.pt > 14) & mu_idiso(events, self._campaign)
        ]

        ## Electron cuts
        iso_ele_ele = events.Electron[
            (events.Electron.pt > 27) & ele_mvatightid(events, self._campaign)
        ]
        iso_ele_mu = events.Electron[
            (events.Electron.pt > 15) & ele_mvatightid(events, self._campaign)
        ]

        ## cross leptons
        req_ele = (ak.count(iso_muon_ele.pt, axis=1) == 1) & (
            ak.count(iso_ele_ele.pt, axis=1) == 1
        )
        req_mu = (ak.count(iso_muon_mu.pt, axis=1) == 1) & (
            ak.count(iso_ele_mu.pt, axis=1) == 1
        )
        iso_ele = ak.concatenate([iso_ele_mu, iso_ele_ele], axis=1)
        iso_mu = ak.concatenate([iso_muon_mu, iso_muon_ele], axis=1)
        iso_ele = ak.pad_none(iso_ele, 1)
        iso_mu = ak.pad_none(iso_mu, 1)

        ## Jet cuts
        event_jet = events.Jet[
            ak.fill_none(
                jet_id(events, self._campaign)
                & (
                    ak.all(
                        events.Jet.metric_table(iso_ele) > 0.4,
                        axis=2,
                        mask_identity=True,
                    )
                )
                & (
                    ak.all(
                        events.Jet.metric_table(iso_mu) > 0.4,
                        axis=2,
                        mask_identity=True,
                    )
                ),
                False,
                axis=-1,
            )
        ]
        req_jets = ak.count(event_jet.pt, axis=1) >= 2

        ## Soft Muon cuts
        soft_muon = events.Muon[softmu_mask(events, self._campaign)]
        req_softmu = ak.count(soft_muon.pt, axis=1) >= 1

        ## Muon jet cuts
        mu_jet = event_jet[
            ak.fill_none(
                (
                    ak.all(
                        event_jet.metric_table(soft_muon) <= 0.4,
                        axis=2,
                        mask_identity=True,
                    )
                )
                & ((event_jet.muonIdx1 != -1) | (event_jet.muonIdx2 != -1)),
                False,
                axis=-1,
            )
        ]
        req_mujet = ak.count(mu_jet.pt, axis=1) >= 1

        ## store jet index for PFCands, create mask on the jet index
        jetindx = ak.mask(
            ak.local_index(events.Jet.pt),
            (
                jet_id(events, self._campaign)
                & (
                    ak.all(
                        events.Jet.metric_table(soft_muon) <= 0.4,
                        axis=2,
                        mask_identity=True,
                    )
                )
                & ((events.Jet.muonIdx1 != -1) | (events.Jet.muonIdx2 != -1))
            )
            == 1,
        )
        jetindx = ak.pad_none(jetindx, 1)
        jetindx = jetindx[:, 0]

        ## Other cuts
        req_dilepmass = ((iso_mu[:, 0] + iso_ele[:, 0]).mass > 12.0) & (
            ((iso_mu[:, 0] + iso_ele[:, 0]).mass < 75)
            | ((iso_mu[:, 0] + iso_ele[:, 0]).mass > 105)
        )

        MET = ak.zip(
            {
                "pt": events.MET.pt,
                "eta": ak.zeros_like(events.MET.pt),
                "phi": events.MET.phi,
                "mass": ak.zeros_like(events.MET.pt),
            },
            with_name="PtEtaPhiMLorentzVector",
        )
        req_MET = MET.pt > 40

        event_level = (
            req_lumi
            & req_MET
            & req_jets
            & req_softmu
            & req_mujet
            & req_dilepmass
            & ((req_trig_ele & req_ele) | (req_trig_mu & req_mu))
        )
        event_level = ak.fill_none(event_level, False)
        if len(events[event_level]) == 0:
            if self.isArray:
                array_writer(
                    self,
                    events[event_level],
                    events,
                    "nominal",
                    dataset,
                    isRealData,
                    empty=True,
                )
            return {dataset: output}

        ####################
        # Selected objects #
        ####################
        shmu = iso_mu[event_level]
        shele = iso_ele[event_level]
        ssmu = soft_muon[event_level]
        nsoftmu = ak.count(ssmu.pt, axis=1)
        softmu0 = ssmu[:, 0]
        sz = shmu[:, 0] + shele[:, 0]
        isomu0 = shmu[:, 0]
        isomu1 = shele[:, 0]
        sjets = event_jet[event_level]
        smuon_jet = mu_jet[event_level]
        nmujet = ak.count(smuon_jet.pt, axis=1)
        smuon_jet = smuon_jet[:, 0]
        smet = MET[event_level]
        njet = ak.count(sjets.pt, axis=1)
        # Find the PFCands associate with selected jets. Search from jetindex->JetPFCands->PFCand
        if "PFCands" in events.fields:
            spfcands = events[event_level].PFCands[
                events[event_level]
                .JetPFCands[
                    events[event_level].JetPFCands.jetIdx == jetindx[event_level]
                ]
                .pFCandsIdx
            ]

        ####################
        # Weight & Geninfo #
        ####################
        weights = Weights(len(events[event_level]), storeIndividual=True)
        if not isRealData:
            weights.add("genweight", events[event_level].genWeight)
            par_flav = (sjets.partonFlavour == 0) & (sjets.hadronFlavour == 0)
            genflavor = ak.values_astype(sjets.hadronFlavour + 1 * par_flav, int)
            smpu = (smuon_jet.partonFlavour == 0) & (smuon_jet.hadronFlavour == 0)
            smflav = ak.values_astype(1 * smpu + smuon_jet.hadronFlavour, int)
            if len(self.SF_map.keys()) > 0:
                syst_wei = True if self.isSyst != False else False
                if "PU" in self.SF_map.keys():
                    puwei(
                        events[event_level].Pileup.nTrueInt,
                        self.SF_map,
                        weights,
                        syst_wei,
                    )
                if "MUO" in self.SF_map.keys():
                    muSFs(isomu0, self.SF_map, weights, syst_wei, False)
                if "EGM" in self.SF_map.keys():
                    eleSFs(isomu1, self.SF_map, weights, syst_wei, False)
                if "BTV" in self.SF_map.keys():
                    btagSFs(sjets, self.SF_map, weights, "DeepJetC", syst_wei)
                    btagSFs(sjets, self.SF_map, weights, "DeepJetB", syst_wei)
                    btagSFs(sjets, self.SF_map, weights, "DeepCSVB", syst_wei)
                    btagSFs(sjets, self.SF_map, weights, "DeepCSVC", syst_wei)
        else:
            genflavor = ak.zeros_like(sjets.pt, dtype=int)
            smflav = ak.zeros_like(smuon_jet.pt, dtype=int)

        # Systematics information
        if shift_name is None:
            systematics = ["nominal"] + list(weights.variations)
        else:
            systematics = [shift_name]
        exclude_btv = [
            "DeepCSVC",
            "DeepCSVB",
            "DeepJetB",
            "DeepJetC",
        ]  # exclude b-tag SFs for btag inputs

        ####################
        #  Fill histogram  #
        ####################
        for syst in systematics:
            if self.isSyst == False and syst != "nominal":
                break
            if self.noHist:
                break
            weight = (
                weights.weight()
                if syst == "nominal" or syst == shift_name
                else weights.weight(modifier=syst)
            )
            for histname, h in output.items():
                if (
                    "Deep" in histname
                    and "btag" not in histname
                    and histname in events.Jet.fields
                ):
                    h.fill(
                        syst,
                        flatten(genflavor),
                        flatten(sjets[histname]),
                        weight=flatten(
                            ak.broadcast_arrays(
                                weights.partial_weight(exclude=exclude_btv), sjets["pt"]
                            )[0]
                        ),
                    )
                elif (
                    "PFCands" in events.fields
                    and "PFCands" in histname
                    and histname.split("_")[1] in events.PFCands.fields
                ):
                    h.fill(
                        syst,
                        flatten(ak.broadcast_arrays(smflav, spfcands["pt"])[0]),
                        flatten(spfcands[histname.replace("PFCands_", "")]),
                        weight=flatten(
                            ak.broadcast_arrays(
                                weights.partial_weight(exclude=exclude_btv),
                                spfcands["pt"],
                            )[0]
                        ),
                    )
                elif "jet_" in histname and "mu" not in histname:
                    h.fill(
                        syst,
                        flatten(genflavor),
                        flatten(sjets[histname.replace("jet_", "")]),
                        weight=flatten(ak.broadcast_arrays(weight, sjets["pt"])[0]),
                    )
                elif "hl_" in histname and histname.replace("hl_", "") in isomu0.fields:
                    h.fill(
                        syst,
                        flatten(isomu0[histname.replace("hl_", "")]),
                        weight=weight,
                    )
                elif "sl_" in histname and histname.replace("sl_", "") in isomu1.fields:
                    h.fill(
                        syst,
                        flatten(isomu1[histname.replace("sl_", "")]),
                        weight=weight,
                    )
                elif "soft_l" in histname and not "ptratio" in histname:
                    h.fill(
                        syst,
                        smflav,
                        flatten(softmu0[histname.replace("soft_l_", "")]),
                        weight=weight,
                    )
                elif "lmujet_" in histname:
                    h.fill(
                        syst,
                        smflav,
                        flatten(smuon_jet[histname.replace("lmujet_", "")]),
                        weight=weight,
                    )
                elif (
                    "btag" in histname
                    and "0" in histname
                    and histname.replace("_0", "") in events.Jet.fields
                ):
                    for i in range(2):
                        if (
                            str(i) not in histname
                            or histname.replace(f"_{i}", "") not in events.Jet.fields
                        ):
                            continue
                        if i == 1 and any(j < 2 for j in njet):
                            continue

                        h.fill(
                            syst="noSF",
                            flav=smflav,
                            discr=smuon_jet[histname.replace(f"_{i}", "")],
                            weight=weights.partial_weight(exclude=exclude_btv),
                        )
                        if not isRealData and "btag" in self.SF_map.keys():
                            h.fill(
                                syst=syst,
                                flav=smflav,
                                discr=smuon_jet[histname.replace(f"_{i}", "")],
                                weight=weight,
                            )

            output["njet"].fill(syst, njet, weight=weight)
            output["nmujet"].fill(syst, nmujet, weight=weight)
            output["nsoftmu"].fill(syst, nsoftmu, weight=weight)
            output["hl_ptratio"].fill(
                syst,
                genflavor[:, 0],
                ratio=isomu0.pt / sjets[:, 0].pt,
                weight=weight,
            )
            output["sl_ptratio"].fill(
                syst,
                genflavor[:, 0],
                ratio=isomu1.pt / sjets[:, 0].pt,
                weight=weight,
            )
            output["soft_l_ptratio"].fill(
                syst,
                flav=smflav,
                ratio=softmu0.pt / smuon_jet.pt,
                weight=weight,
            )
            output["dr_lmujetsmu"].fill(
                syst,
                flav=smflav,
                dr=smuon_jet.delta_r(softmu0),
                weight=weight,
            )
            output["dr_lmujethmu"].fill(
                syst,
                flav=smflav,
                dr=smuon_jet.delta_r(isomu0),
                weight=weight,
            )
            output["dr_lmusmu"].fill(syst, isomu0.delta_r(softmu0), weight=weight)
            output["z_pt"].fill(syst, flatten(sz.pt), weight=weight)
            output["z_eta"].fill(syst, flatten(sz.eta), weight=weight)
            output["z_phi"].fill(syst, flatten(sz.phi), weight=weight)
            output["z_mass"].fill(syst, flatten(sz.mass), weight=weight)
            output["MET_pt"].fill(syst, flatten(smet.pt), weight=weight)
            output["MET_phi"].fill(syst, flatten(smet.phi), weight=weight)
            output["npvs"].fill(
                syst,
                events[event_level].PV.npvs,
                weight=weight,
            )
            if not isRealData:
                output["pu"].fill(
                    syst,
                    events[event_level].Pileup.nTrueInt,
                    weight=weight,
                )
        #######################
        #  Create root files  #
        #######################
        if self.isArray:
            # Keep the structure of events and pruned the object size
            pruned_ev = events[event_level]
            pruned_ev["SelJet"] = sjets
            pruned_ev["Muon"] = isomu0
            pruned_ev["Electron"] = isomu1
            pruned_ev["MuonJet"] = smuon_jet
            pruned_ev["SoftMuon"] = ssmu[:, 0]
            pruned_ev["dilep"] = isomu0[:, 0] + isomu1[:, 1]
            pruned_ev["dilep", "pt"] = pruned_ev.dilep.pt
            pruned_ev["dilep", "eta"] = pruned_ev.dilep.eta
            pruned_ev["dilep", "phi"] = pruned_ev.dilep.phi
            pruned_ev["dilep", "mass"] = pruned_ev.dilep.mass
            if "PFCands" in events.fields:
                pruned_ev.PFCands = spfcands
            # Add custom variables
            if not isRealData:
                pruned_ev["weight"] = weights.weight()
                for ind_wei in weights.weightStatistics.keys():
                    pruned_ev[f"{ind_wei}_weight"] = weights.partial_weight(
                        include=[ind_wei]
                    )
            pruned_ev["dr_mujet_softmu"] = pruned_ev.SoftMuon.delta_r(smuon_jet)
            pruned_ev["dr_mujet_lep1"] = pruned_ev.Muon.delta_r(smuon_jet)
            pruned_ev["dr_mujet_lep2"] = pruned_ev.Electron.delta_r(smuon_jet)
            pruned_ev["dr_lep1_softmu"] = pruned_ev.Muon.delta_r(pruned_ev.SoftMuon)
            pruned_ev["soft_l_ptratio"] = pruned_ev.SoftMuon.pt / smuon_jet.pt
            pruned_ev["l1_ptratio"] = pruned_ev.Muon.pt / smuon_jet.pt
            pruned_ev["l2_ptratio"] = pruned_ev.Electron.pt / smuon_jet.pt
            array_writer(self, pruned_ev, events, systematics[0], dataset, isRealData)

        return {dataset: output}

    def postprocess(self, accumulator):
        return accumulator

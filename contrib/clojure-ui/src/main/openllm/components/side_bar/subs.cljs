(ns openllm.components.side-bar.subs
  (:require [openllm.components.subs :as components-subs]
            [openllm.components.side-bar.db :as db]
            [re-frame.core :refer [reg-sub]]))

(def parameter-id->human-readable
  {::db/temperature "Temperature"
   ::db/top_k "Top K"
   ::db/top_p "Top P"
   ::db/typical_p "Typical P"
   ::db/epsilon_cutoff "Epsilon Cutoff"
   ::db/eta_cutoff "Eta Cutoff"
   ::db/diversity_penalty "Diversity Penalty"
   ::db/repetition_penalty "Repetition Penalty"
   ::db/encoder_repetition_penalty "Encoder Repetition Penalty"
   ::db/length_penalty "Length Penalty"
   ::db/max_new_tokens "Maximum New Tokens"
   ::db/min_length "Minimum Length"
   ::db/min_new_tokens "Minimum New Tokens"
   ::db/early_stopping "Early Stopping"
   ::db/max_time "Maximum Time"
   ::db/num_beams "Number of Beams"
   ::db/num_beam_groups "Number of Beam Groups"
   ::db/penalty_alpha "Penalty Alpha"
   ::db/use_cache "Use Cache"})

(reg-sub
 ::side-bar-open?
 :<- [::components-subs/side-bar-db]
 (fn [side-bar-db _]
   (:side-bar-open? side-bar-db)))

(reg-sub
 ::model-config
 :<- [::components-subs/side-bar-db]
 (fn [side-bar-db _]
   (:model-config side-bar-db)))

(reg-sub
 ::human-readable-config
 :<- [::model-config]
 (fn [model-config _]
   (vec (map (fn [[k v]]
               [k {:name (parameter-id->human-readable k)
                   :value v}])
             model-config))))

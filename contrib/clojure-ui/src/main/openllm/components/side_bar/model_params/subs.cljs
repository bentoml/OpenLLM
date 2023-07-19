(ns openllm.components.side-bar.model-params.subs
  (:require [openllm.components.side-bar.subs :as side-bar-subs]
            [re-frame.core :refer [reg-sub]]))

(def parameter-id->human-readable
  "Maps paramter ids to human readable names."
  {:temperature "Temperature"
   :top_k "Top K"
   :top_p "Top P"
   :typical_p "Typical P"
   :epsilon_cutoff "Epsilon Cutoff"
   :eta_cutoff "Eta Cutoff"
   :diversity_penalty "Diversity Penalty"
   :repetition_penalty "Repetition Penalty"
   :encoder_repetition_penalty "Encoder Repetition Penalty"
   :length_penalty "Length Penalty"
   :max_new_tokens "Maximum New Tokens"
   :min_length "Minimum Length"
   :min_new_tokens "Minimum New Tokens"
   :early_stopping "Early Stopping"
   :max_time "Maximum Time"
   :num_beams "Number of Beams"
   :num_beam_groups "Number of Beam Groups"
   :penalty_alpha "Penalty Alpha"
   :use_cache "Use Cache"})

(reg-sub
 ::model-config
 :<- [::side-bar-subs/model-params-db]
 (fn [model-params-db _]
   model-params-db))

(reg-sub
 ::human-readable-config
 :<- [::model-config]
 (fn [model-config _]
   (vec (map (fn [[k v]]
               [k {:name (parameter-id->human-readable k)
                   :value v}])
             model-config))))

import requests
from typing import List, Dict, Tuple, Any
#from rcsbsearchapi import TextQuery, AttributeQuery  # deprecated
from rcsbapi.search import TextQuery, AttributeQuery


class PDBQuery:
    POL_URL = 'https://data.rcsb.org/rest/v1/core/polymer_entity/{entry_id}/{entity_id}'

    def __init__(self):
        pass

    def get_pdb_ids_by_gene(self, query: str, organism: str, gene_name: str) -> Tuple[List[str], List[str]]:
        raw_results = self._search_entities(query=query, organism=organism)
        # print("Raw results:", raw_results)
        clean_list, flag_list = self._filter_entities(raw_results, gene_name=gene_name, organism=organism)
        return clean_list, flag_list

    def bulk_gene_queries(self, query_list: List[str], org_list: List[str], gene_list: List[str]) -> Dict[str, Any]:
        """Run gene → PDB lookups for multiple entries and return structured results keyed by gene-organism."""
        if not (len(query_list) == len(org_list) == len(gene_list)):
            raise ValueError("query_list, org_list, and gene_list must be the same length.")

        results = {}

        for query, org, gene in zip(query_list, org_list, gene_list):
            print(f"Running search for {query}, {org}, {gene}")
            pdb_ids, flags = self.get_pdb_ids_by_gene(query=query, organism=org, gene_name=gene)

            # Create a unique key like "taar9_mouse" or "TAAR1_HUMAN"
            key = f"{gene.strip().lower()}_{'human' if 'homo sapiens' in org.lower() else 'mouse'}"

            results[key] = {
                "pdb_ids": [pdb_id.split("_")[0] for pdb_id in pdb_ids],
                "flags": flags
            }

        return results

    def _search_entities(self, query: str, organism: str):
        """Loosened filters to improve recall of relevant entries"""
        q1 = TextQuery(query)
        q2 = AttributeQuery("rcsb_entity_source_organism.ncbi_scientific_name", "exact_match", organism)
        q3 = AttributeQuery("entity_poly.rcsb_entity_polymer_type", "exact_match", "Protein")

        combined_query = q1 & q2 & q3
        return list(combined_query("polymer_entity"))

    def _filter_entities(self, search_results: List[str], organism: str, gene_name: str) -> Tuple[List[str], List[str]]:
        clean_list_org = []
        clean_list_gene = []
        flagged = []

        for item in search_results:
            entry_id, entity_id = item.split('_')
            try:
                response = requests.get(self.POL_URL.format(entry_id=entry_id, entity_id=entity_id))
                response.raise_for_status()
                data = response.json()

                # Organism check
                source_organisms = data.get('rcsb_entity_source_organism', [{}])
                org_match = any(org.get('ncbi_scientific_name', '').lower() == organism.lower()
                                for org in source_organisms)

                if not org_match:
                    continue
                clean_list_org.append(entry_id)

                # Gene name check
                gene_info = []

                # Method 1: entity_src_gen.pdbx_gene_src_gene
                if 'entity_src_gen' in data:
                    for g in data['entity_src_gen']:
                        if 'pdbx_gene_src_gene' in g:
                            gene_info += g['pdbx_gene_src_gene'].split(',')

                # Method 2: rcsb_polymer_entity.gene_name
                gene_info += data.get('rcsb_polymer_entity', {}).get('gene_name', [])

                gene_info = [g.strip().lower() for g in gene_info if g]

                if gene_name.lower() in gene_info:
                    clean_list_gene.append(entry_id)

            except Exception as e:
                flagged.append(f"{item} failed: {e}")

        # Keep only those that matched both
        clean_combined = list(set(clean_list_org) & set(clean_list_gene))
        return sorted(set(clean_combined)), flagged

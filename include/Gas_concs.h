#ifndef GAS_CONCS_H
#define GAS_CONCS_H

#include <vector>

template<typename TF>
class Gas_concs
{
    public:
        Gas_concs() {}
        Gas_concs(const Gas_concs& gas_concs_ref, const int start, const int size)
        {
            const int end = start + size - 1;
            for (auto& g : gas_concs_ref.gas_concs_map)
            {
                Array<TF,2> gas_conc_subset = g.second.subset({{ {start, end}, {1, g.second.dim(2)} }});
                this->gas_concs_map.emplace(g.first, gas_conc_subset);
            }
        }
        // Insert new gas into the map.
        void set_vmr(const std::string& name, const Array<TF,2>& data)
        {
            gas_concs_map.emplace(name, data);
        }

        // Insert new gas into the map.
        void get_vmr(const std::string& name, Array<TF,2>& data) const
        {
            data = this->gas_concs_map.at(name);
        }

        // Check if gas exists in map.
        bool exists(const std::string& name) const { return gas_concs_map.count(name) != 0; }

    private:
        std::map<std::string, Array<TF,2>> gas_concs_map;
};
#endif

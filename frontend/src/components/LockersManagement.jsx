import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { toast } from "sonner";
import axios from "axios";
import { Save, Shirt } from "lucide-react";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const SIZES = ['XS', 'S', 'M', 'L', 'XL', 'XXL'];

export default function LockersManagement({ swimmers }) {
  const [selectedSwimmer, setSelectedSwimmer] = useState('');
  const [lockerData, setLockerData] = useState({
    pants_size: '',
    shirt_size: '',
    hoodie_size: ''
  });
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (selectedSwimmer) {
      fetchLocker(selectedSwimmer);
    }
  }, [selectedSwimmer]);

  const fetchLocker = async (swimmerId) => {
    try {
      const token = localStorage.getItem('token');
      const headers = { Authorization: `Bearer ${token}` };
      const response = await axios.get(`${API}/lockers/${swimmerId}`, { headers });
      setLockerData({
        pants_size: response.data.pants_size,
        shirt_size: response.data.shirt_size,
        hoodie_size: response.data.hoodie_size
      });
    } catch (error) {
      // Locker doesn't exist yet
      setLockerData({
        pants_size: '',
        shirt_size: '',
        hoodie_size: ''
      });
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const token = localStorage.getItem('token');
      const headers = { Authorization: `Bearer ${token}` };
      
      await axios.post(`${API}/lockers`, {
        swimmer_id: selectedSwimmer,
        ...lockerData
      }, { headers });

      toast.success('Taquilla actualizada correctamente');
    } catch (error) {
      toast.error(error.response?.data?.detail || 'Error al actualizar taquilla');
    } finally {
      setLoading(false);
    }
  };

  if (swimmers.length === 0) {
    return (
      <div className="text-center py-12 text-gray-500">
        <p className="text-lg">No hay nadadores registrados</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <Label>Seleccionar Nadador</Label>
        <Select value={selectedSwimmer} onValueChange={setSelectedSwimmer}>
          <SelectTrigger className="w-full" data-testid="swimmer-select-locker">
            <SelectValue placeholder="Seleccionar un nadador" />
          </SelectTrigger>
          <SelectContent>
            {swimmers.map((swimmer) => (
              <SelectItem key={swimmer.id} value={swimmer.id}>
                {swimmer.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {selectedSwimmer && (
        <Card className="bg-gradient-to-br from-emerald-50 to-teal-50 border-[#278D33]/20">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-[#278D33]">
              <Shirt className="w-5 h-5" />
              Configurar Taquilla
            </CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="space-y-2">
                  <Label htmlFor="pants">Talla de Pantalón</Label>
                  <Select 
                    value={lockerData.pants_size} 
                    onValueChange={(value) => setLockerData({...lockerData, pants_size: value})}
                  >
                    <SelectTrigger data-testid="pants-size-select">
                      <SelectValue placeholder="Seleccionar talla" />
                    </SelectTrigger>
                    <SelectContent>
                      {SIZES.map((size) => (
                        <SelectItem key={size} value={size}>
                          {size}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="shirt">Talla de Camiseta</Label>
                  <Select 
                    value={lockerData.shirt_size} 
                    onValueChange={(value) => setLockerData({...lockerData, shirt_size: value})}
                  >
                    <SelectTrigger data-testid="shirt-size-select">
                      <SelectValue placeholder="Seleccionar talla" />
                    </SelectTrigger>
                    <SelectContent>
                      {SIZES.map((size) => (
                        <SelectItem key={size} value={size}>
                          {size}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="hoodie">Talla de Sudadera</Label>
                  <Select 
                    value={lockerData.hoodie_size} 
                    onValueChange={(value) => setLockerData({...lockerData, hoodie_size: value})}
                  >
                    <SelectTrigger data-testid="hoodie-size-select">
                      <SelectValue placeholder="Seleccionar talla" />
                    </SelectTrigger>
                    <SelectContent>
                      {SIZES.map((size) => (
                        <SelectItem key={size} value={size}>
                          {size}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <Button 
                type="submit" 
                disabled={loading || !lockerData.pants_size || !lockerData.shirt_size || !lockerData.hoodie_size}
                className="w-full bg-[#278D33] hover:bg-[#1f6b28]"
                data-testid="save-locker-button"
              >
                <Save className="w-4 h-4 mr-2" />
                {loading ? 'Guardando...' : 'Guardar Taquilla'}
              </Button>
            </form>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

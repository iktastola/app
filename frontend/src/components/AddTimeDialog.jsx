import { useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

const DISTANCES = [50, 100, 200, 400, 800, 1500];
const STYLES = ['Libre', 'Espalda', 'Braza', 'Mariposa'];

export default function AddTimeDialog({ open, onOpenChange, onSubmit, swimmers }) {
  const [formData, setFormData] = useState({
    swimmer_id: '',
    distance: '',
    style: '',
    minutes: '',
    seconds: '',
    date: new Date().toISOString().split('T')[0],
    competition: ''
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    
    const totalSeconds = (parseInt(formData.minutes) || 0) * 60 + parseFloat(formData.seconds || 0);
    
    const timeData = {
      swimmer_id: formData.swimmer_id,
      distance: parseInt(formData.distance),
      style: formData.style,
      time_seconds: totalSeconds,
      date: new Date(formData.date).toISOString(),
      competition: formData.competition || null
    };
    
    onSubmit(timeData);
    setFormData({
      swimmer_id: '',
      distance: '',
      style: '',
      minutes: '',
      seconds: '',
      date: new Date().toISOString().split('T')[0],
      competition: ''
    });
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md" data-testid="add-time-dialog">
        <DialogHeader>
          <DialogTitle className="text-2xl text-[#278D33]">Registrar Nuevo Tiempo</DialogTitle>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="swimmer">Nadador</Label>
            <Select value={formData.swimmer_id} onValueChange={(value) => setFormData({...formData, swimmer_id: value})}>
              <SelectTrigger data-testid="swimmer-select">
                <SelectValue placeholder="Seleccionar nadador" />
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

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="distance">Distancia</Label>
              <Select value={formData.distance.toString()} onValueChange={(value) => setFormData({...formData, distance: value})}>
                <SelectTrigger data-testid="distance-select">
                  <SelectValue placeholder="Metros" />
                </SelectTrigger>
                <SelectContent>
                  {DISTANCES.map((dist) => (
                    <SelectItem key={dist} value={dist.toString()}>
                      {dist}m
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="style">Estilo</Label>
              <Select value={formData.style} onValueChange={(value) => setFormData({...formData, style: value})}>
                <SelectTrigger data-testid="style-select">
                  <SelectValue placeholder="Estilo" />
                </SelectTrigger>
                <SelectContent>
                  {STYLES.map((style) => (
                    <SelectItem key={style} value={style}>
                      {style}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="space-y-2">
            <Label>Tiempo</Label>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Input
                  type="number"
                  placeholder="Minutos"
                  value={formData.minutes}
                  onChange={(e) => setFormData({...formData, minutes: e.target.value})}
                  min="0"
                  data-testid="minutes-input"
                />
              </div>
              <div>
                <Input
                  type="number"
                  step="0.01"
                  placeholder="Segundos"
                  value={formData.seconds}
                  onChange={(e) => setFormData({...formData, seconds: e.target.value})}
                  min="0"
                  max="59.99"
                  required
                  data-testid="seconds-input"
                />
              </div>
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="date">Fecha</Label>
            <Input
              type="date"
              value={formData.date}
              onChange={(e) => setFormData({...formData, date: e.target.value})}
              required
              data-testid="date-input"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="competition">Competición (opcional)</Label>
            <Input
              placeholder="Nombre de la competición"
              value={formData.competition}
              onChange={(e) => setFormData({...formData, competition: e.target.value})}
              data-testid="competition-input"
            />
          </div>

          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>
              Cancelar
            </Button>
            <Button type="submit" className="bg-[#278D33] hover:bg-[#1f6b28]" data-testid="submit-time-button">
              Registrar Tiempo
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

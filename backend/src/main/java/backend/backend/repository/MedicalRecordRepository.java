package backend.backend.repository;

import backend.backend.entity.MedicalRecord;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;
import java.util.List;

@Repository
public interface MedicalRecordRepository extends JpaRepository<MedicalRecord, Long> {
    @Query("SELECT mr FROM MedicalRecord mr LEFT JOIN FETCH mr.prediction WHERE mr.patient.id = :patientId ORDER BY mr.createdAt DESC")
    List<MedicalRecord> findByPatientIdOrderByCreatedAtDesc(@Param("patientId") Long patientId);
} 